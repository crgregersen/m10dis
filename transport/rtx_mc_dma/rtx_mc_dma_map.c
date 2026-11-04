// Map active M10-local PEX8747 multicast ingress windows in the RTX DMA
// domain. This creates translations only; it does not issue PCIe payloads or
// change PCI configuration. Successful maps live until module removal.
#include <linux/dma-mapping.h>
#include <linux/module.h>
#include <linux/pci.h>
#include <linux/errno.h>
#include <linux/kernel.h>
#include <linux/string.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/dma-mapping.h>

#define MAX_GROUPS 6
#define WINDOW_SIZE SZ_64K
#define HOST_TEST_SIZE SZ_64K

/* No default: the loader must pass the RTX address discovered at run time. */
static char *requester_bdf;
module_param(requester_bdf, charp, 0444);
MODULE_PARM_DESC(requester_bdf, "RTX requester PCI BDF (required, from qwen38_pci_discover.py)");

static int groups = MAX_GROUPS;
module_param(groups, int, 0444);
MODULE_PARM_DESC(groups, "Number of active card-local PEX groups (1..6)");

static unsigned long long ingress[MAX_GROUPS];
static int ingress_count;
module_param_array(ingress, ullong, &ingress_count, 0444);
MODULE_PARM_DESC(ingress, "Live PEX ingress resource addresses");

static unsigned long long branch_start[MAX_GROUPS];
static int branch_start_count;
module_param_array(branch_start, ullong, &branch_start_count, 0444);
MODULE_PARM_DESC(branch_start, "Local PEX bridge-window starts");

static unsigned long long branch_end[MAX_GROUPS];
static int branch_end_count;
module_param_array(branch_end, ullong, &branch_end_count, 0444);
MODULE_PARM_DESC(branch_end, "Local PEX bridge-window ends");

static unsigned long long peer_resource;
module_param(peer_resource, ullong, 0444);
MODULE_PARM_DESC(peer_resource, "Optional M10 BAR resource for a direct one-GPU control mapping");

static unsigned long long iova0, iova1, iova2, iova3, iova4, iova5;
module_param(iova0, ullong, 0444);
module_param(iova1, ullong, 0444);
module_param(iova2, ullong, 0444);
module_param(iova3, ullong, 0444);
module_param(iova4, ullong, 0444);
module_param(iova5, ullong, 0444);
MODULE_PARM_DESC(iova0, "Read-only DMA address returned for group 0");
MODULE_PARM_DESC(iova1, "Read-only DMA address returned for group 1");
MODULE_PARM_DESC(iova2, "Read-only DMA address returned for group 2");
MODULE_PARM_DESC(iova3, "Read-only DMA address returned for group 3");
MODULE_PARM_DESC(iova4, "Read-only DMA address returned for group 4");
MODULE_PARM_DESC(iova5, "Read-only DMA address returned for group 5");
static unsigned long long peer_iova;
module_param(peer_iova, ullong, 0444);
MODULE_PARM_DESC(peer_iova, "Read-only DMA address returned for the optional M10 BAR control map");

static void set_iova(unsigned int group, dma_addr_t dma)
{
	switch (group) {
	case 0: iova0 = dma; break;
	case 1: iova1 = dma; break;
	case 2: iova2 = dma; break;
	case 3: iova3 = dma; break;
	case 4: iova4 = dma; break;
	case 5: iova5 = dma; break;
	}
}
static void clear_iova(unsigned int group)
{
	set_iova(group, 0);
}

struct live_map {
	phys_addr_t resource;
	dma_addr_t dma;
	bool active;
};

static struct pci_dev *requester;
static struct live_map maps[MAX_GROUPS];
static struct live_map peer_map;
static void *host_test_cpu;
static dma_addr_t host_test_dma;
static unsigned long long host_iova;
module_param(host_iova, ullong, 0444);
MODULE_PARM_DESC(host_iova, "Requester DMA address of the disposable coherent-RAM control buffer");

static int host_test_mmap(struct file *file, struct vm_area_struct *vma)
{
	if (!host_test_cpu || vma->vm_pgoff != 0 ||
	    vma->vm_end - vma->vm_start != HOST_TEST_SIZE)
		return -EINVAL;
	return dma_mmap_coherent(&requester->dev, vma, host_test_cpu,
				 host_test_dma, HOST_TEST_SIZE);
}

static const struct file_operations host_test_fops = {
	.owner = THIS_MODULE,
	.mmap = host_test_mmap,
};

static struct miscdevice host_test_misc = {
	.minor = MISC_DYNAMIC_MINOR,
	.name = "rtx_mc_dma_map",
	.fops = &host_test_fops,
};

static void release_host_test(void)
{
	if (host_test_cpu && requester) {
		misc_deregister(&host_test_misc);
		dma_free_coherent(&requester->dev, HOST_TEST_SIZE,
				  host_test_cpu, host_test_dma);
		host_test_cpu = NULL;
		host_test_dma = 0;
		host_iova = 0;
	}
}

static int parse_bdf(const char *s, unsigned int *domain, unsigned int *bus,
		     unsigned int *dev, unsigned int *fn)
{
	return sscanf(s, "%x:%x:%x.%x", domain, bus, dev, fn) == 4 ? 0 : -EINVAL;
}

static void release_maps(void)
{
	int i;

	if (!requester)
		return;
	if (peer_map.active) {
		dma_unmap_resource(&requester->dev, peer_map.dma, WINDOW_SIZE,
				   DMA_FROM_DEVICE, 0);
		pr_info("rtx_mc_dma_map: PEER_UNMAP iova=%pad len=%u dir=DMA_FROM_DEVICE\n",
			&peer_map.dma, WINDOW_SIZE);
		peer_map.active = false;
		peer_iova = 0;
	}
	for (i = MAX_GROUPS - 1; i >= 0; --i) {
		if (!maps[i].active)
			continue;
		dma_unmap_resource(&requester->dev, maps[i].dma, WINDOW_SIZE,
				   DMA_FROM_DEVICE, 0);
		pr_info("rtx_mc_dma_map: UNMAP group=%d iova=%pad len=%u dir=DMA_FROM_DEVICE\n",
			i, &maps[i].dma, WINDOW_SIZE);
		maps[i].active = false;
		clear_iova(i);
	}
}

static int __init rtx_mc_dma_map_init(void)
{
	unsigned int domain, bus, dev, fn;
	int i, j;

	if (groups < 1 || groups > MAX_GROUPS || ingress_count != groups ||
	    branch_start_count != groups || branch_end_count != groups)
		return -EINVAL;
	if (!requester_bdf || !*requester_bdf) {
		pr_err("rtx_mc_dma_map: requester_bdf is required\n");
		return -EINVAL;
	}
	if (parse_bdf(requester_bdf, &domain, &bus, &dev, &fn))
		return -EINVAL;
	requester = pci_get_domain_bus_and_slot(domain, bus, PCI_DEVFN(dev, fn));
	if (!requester)
		return -ENODEV;
	if (requester->vendor != PCI_VENDOR_ID_NVIDIA || requester->device != 0x2482) {
		pr_err("rtx_mc_dma_map: requester %s is %04x:%04x, expected RTX 3070 Ti 10de:2482\n",
		       requester_bdf, requester->vendor, requester->device);
		goto invalid;
	}
	if (!requester->dev.dma_mask || !*requester->dev.dma_mask)
		goto invalid;
	pr_info("rtx_mc_dma_map: requester=%s id=%04x:%04x dma_mask=%#llx coherent_mask=%#llx groups=%d direction=DMA_FROM_DEVICE\n",
		requester_bdf, requester->vendor, requester->device,
		(unsigned long long)*requester->dev.dma_mask,
		(unsigned long long)requester->dev.coherent_dma_mask, groups);

	for (i = 0; i < groups; ++i) {
		phys_addr_t start = (phys_addr_t)ingress[i];
		phys_addr_t end = start + WINDOW_SIZE - 1;
		dma_addr_t dma;

		if (!start || (start & (WINDOW_SIZE - 1)) || end < start ||
		    start < branch_start[i] || end > branch_end[i]) {
			pr_err("rtx_mc_dma_map: invalid group=%d ingress=%pa bridge=%#llx-%#llx\n",
			       i, &start, branch_start[i], branch_end[i]);
			goto invalid_maps;
		}
		for (j = 0; j < i; ++j) {
			if (ingress[j] == ingress[i]) {
				pr_err("rtx_mc_dma_map: duplicate ingress groups=%d,%d\n", j, i);
				goto invalid_maps;
			}
		}
		dma = dma_map_resource(&requester->dev, start, WINDOW_SIZE,
				       DMA_FROM_DEVICE, 0);
		if (dma_mapping_error(&requester->dev, dma)) {
			pr_err("rtx_mc_dma_map: MAP_FAILED group=%d requester=%s resource=%pa len=%u dir=DMA_FROM_DEVICE\n",
			       i, requester_bdf, &start, WINDOW_SIZE);
			goto failed_maps;
		}
		maps[i].resource = start;
		maps[i].dma = dma;
		maps[i].active = true;
		set_iova(i, dma);
		pr_info("rtx_mc_dma_map: MAP_OK group=%d requester=%s resource=%pa len=%u dir=DMA_FROM_DEVICE iova=%pad iova_end=%pad\n",
			i, requester_bdf, &maps[i].resource, WINDOW_SIZE,
			&maps[i].dma, &(dma_addr_t){dma + WINDOW_SIZE - 1});
	}
	if (peer_resource) {
		phys_addr_t start = (phys_addr_t)peer_resource;
		phys_addr_t end = start + WINDOW_SIZE - 1;
		dma_addr_t dma;
		if ((start & (WINDOW_SIZE - 1)) || end < start ||
		    start < branch_start[0] || end > branch_end[0] ||
		    (start < ingress[0] + WINDOW_SIZE && ingress[0] < start + WINDOW_SIZE)) {
			pr_err("rtx_mc_dma_map: invalid optional peer resource=%pa bridge=%#llx-%#llx\n",
			       &start, branch_start[0], branch_end[0]);
			goto failed_maps;
		}
		dma = dma_map_resource(&requester->dev, start, WINDOW_SIZE,
				       DMA_FROM_DEVICE, 0);
		if (dma_mapping_error(&requester->dev, dma)) {
			pr_err("rtx_mc_dma_map: PEER_MAP_FAILED requester=%s resource=%pa len=%u dir=DMA_FROM_DEVICE\n",
			       requester_bdf, &start, WINDOW_SIZE);
			goto failed_maps;
		}
		peer_map.resource = start;
		peer_map.dma = dma;
		peer_map.active = true;
		peer_iova = dma;
		pr_info("rtx_mc_dma_map: PEER_MAP_OK requester=%s resource=%pa len=%u dir=DMA_FROM_DEVICE iova=%pad\n",
			requester_bdf, &peer_map.resource, WINDOW_SIZE, &peer_map.dma);
	}
	host_test_cpu = dma_alloc_coherent(&requester->dev, HOST_TEST_SIZE,
					   &host_test_dma, GFP_KERNEL);
	if (!host_test_cpu) {
		pr_err("rtx_mc_dma_map: coherent host test allocation failed size=%u\n",
		       HOST_TEST_SIZE);
		goto failed_maps;
	}
	memset(host_test_cpu, 0, HOST_TEST_SIZE);
	host_iova = host_test_dma;
	if (misc_register(&host_test_misc)) {
		pr_err("rtx_mc_dma_map: coherent host test device registration failed\n");
		dma_free_coherent(&requester->dev, HOST_TEST_SIZE,
				  host_test_cpu, host_test_dma);
		host_test_cpu = NULL;
		host_test_dma = 0;
		host_iova = 0;
		goto failed_maps;
	}
	pr_info("rtx_mc_dma_map: HOST_TEST_OK requester=%s iova=%pad len=%u cpu_mapping=/dev/%s prot=coherent-RW\n",
		requester_bdf, &host_test_dma, HOST_TEST_SIZE, host_test_misc.name);
	return 0;

invalid_maps:
	release_maps();
invalid:
	pci_dev_put(requester);
	requester = NULL;
	return -EINVAL;
failed_maps:
	release_host_test();
	release_maps();
	pci_dev_put(requester);
	requester = NULL;
	return -EIO;
}

static void __exit rtx_mc_dma_map_exit(void)
{
	release_host_test();
	release_maps();
	if (requester) {
		pci_dev_put(requester);
		requester = NULL;
	}
	pr_info("rtx_mc_dma_map: all successfully acquired DMA mappings released\n");
}

module_init(rtx_mc_dma_map_init);
module_exit(rtx_mc_dma_map_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Codex diagnostic");
MODULE_DESCRIPTION("Bounded RTX DMA mappings for active M10-local PEX ingress windows");
