#include <linux/capability.h>
#include <linux/fs.h>
#include <linux/miscdevice.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/printk.h>

#if !defined(QWEN_MC_BASE_0) || !defined(QWEN_MC_BASE_1) || \
    !defined(QWEN_MC_BASE_2) || !defined(QWEN_MC_BASE_3) || \
    !defined(QWEN_MC_BASE_4) || !defined(QWEN_MC_BASE_5)
#error "Build with six freshly audited QWEN_MC_BASE_n addresses"
#endif

#define QWEN_MC_BYTES (64UL * 1024UL)
static const phys_addr_t qwen_mc_phys[6] = {
    QWEN_MC_BASE_0, QWEN_MC_BASE_1, QWEN_MC_BASE_2,
    QWEN_MC_BASE_3, QWEN_MC_BASE_4, QWEN_MC_BASE_5,
};

static int qwen_mc_open(struct inode *inode, struct file *file)
{
    if (!capable(CAP_SYS_RAWIO))
        return -EPERM;
    return nonseekable_open(inode, file);
}

static int qwen_mc_mmap(struct file *file, struct vm_area_struct *vma)
{
    const unsigned long pages_per_window = QWEN_MC_BYTES >> PAGE_SHIFT;
    const unsigned long bytes = vma->vm_end - vma->vm_start;
    unsigned long slot;

    if (!capable(CAP_SYS_RAWIO))
        return -EPERM;
    if (bytes != QWEN_MC_BYTES || vma->vm_pgoff % pages_per_window != 0)
        return -EINVAL;
    slot = vma->vm_pgoff / pages_per_window;
    if (slot >= ARRAY_SIZE(qwen_mc_phys))
        return -EINVAL;
    if (!(vma->vm_flags & VM_WRITE) || (vma->vm_flags & VM_EXEC))
        return -EPERM;
    if (!IS_ALIGNED(qwen_mc_phys[slot], QWEN_MC_BYTES))
        return -EINVAL;

    vm_flags_set(vma, VM_IO | VM_PFNMAP | VM_DONTEXPAND | VM_DONTDUMP);
    vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
    return remap_pfn_range(vma, vma->vm_start,
                           qwen_mc_phys[slot] >> PAGE_SHIFT, bytes,
                           vma->vm_page_prot);
}

static const struct file_operations qwen_mc_fops = {
    .owner = THIS_MODULE, .open = qwen_mc_open, .mmap = qwen_mc_mmap,
};
static struct miscdevice qwen_mc_device = {
    .minor = MISC_DYNAMIC_MINOR, .name = "m10_qwen_mc_wc6",
    .fops = &qwen_mc_fops, .mode = 0600,
};

static int __init qwen_mc_init(void)
{
    int status = misc_register(&qwen_mc_device);
    if (!status)
        pr_info("m10_qwen_mc_wc6: six bounded 64 KiB WC apertures registered\n");
    return status;
}
static void __exit qwen_mc_exit(void) { misc_deregister(&qwen_mc_device); }
module_init(qwen_mc_init);
module_exit(qwen_mc_exit);
MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("Bounded WC mappings for six freshly audited Qwen M10 PEX apertures");
