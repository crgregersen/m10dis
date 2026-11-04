#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/ioctl.h>

typedef struct {
    uint32_t h_root;
    uint32_t h_parent;
    uint32_t h_new;
    uint32_t h_class;
    uint32_t flags;
    uint32_t padding;
    uint64_t memory;
    uint64_t limit;
    uint32_t status;
    uint32_t tail_padding;
    int fd;
    uint32_t wrapper_padding;
} nvos02_with_fd;

typedef struct {
    uint32_t h_root;
    uint32_t h_parent;
    uint32_t h_new;
    uint32_t h_class;
    uint64_t alloc_params;
    uint64_t rights_requested;
    uint32_t params_size;
    uint32_t flags;
    uint32_t status;
    uint32_t padding;
} nvos64;

typedef struct {
    uint32_t h_client;
    uint32_t h_object;
    uint32_t command;
    uint32_t flags;
    uint64_t params;
    uint32_t params_size;
    uint32_t status;
} nvos54;

typedef struct {
    uint32_t type;
    uint32_t h_device;
    uint32_t h_parent;
    uint32_t h_object;
    int32_t fd;
    uint32_t flags;
} export_object_to_fd_params;

typedef struct {
    int32_t fd;
    uint32_t h_device;
    uint16_t max_objects;
    uint8_t metadata[64];
    uint16_t alignment_padding;
    uint32_t objects[512];
    uint16_t num_objects;
    uint16_t index;
} export_objects_to_fd_params;

typedef struct {
    uint32_t h_client;
    uint32_t h_device;
    uint32_t h_memory;
    uint32_t alignment_padding;
    uint64_t offset;
    uint64_t length;
    uint64_t linear_address;
    uint32_t status;
    uint32_t flags;
    int32_t fd;
    uint32_t wrapper_padding;
} nvos33_with_fd;

static uint32_t last_export_client;
static uint32_t last_export_device;
static uint32_t last_export_object;
static int last_export_control_fd = -1;

__attribute__((visibility("default")))
int rm_trace_get_last_export(uint32_t *client, uint32_t *device, uint32_t *object,
                             int *control_fd)
{
    if (!last_export_client || !last_export_device || !last_export_object)
        return 0;
    *client = last_export_client;
    *device = last_export_device;
    *object = last_export_object;
    *control_fd = last_export_control_fd;
    return 1;
}

int ioctl(int fd, unsigned long request, ...)
{
    static int (*real_ioctl)(int, unsigned long, void *);
    void *argument;
    va_list args;

    if (!real_ioctl)
        real_ioctl = dlsym(RTLD_NEXT, "ioctl");

    va_start(args, request);
    argument = va_arg(args, void *);
    va_end(args);

    const unsigned int type = _IOC_TYPE(request);
    const unsigned int number = _IOC_NR(request);
    const unsigned int size = _IOC_SIZE(request);
    if (type == 'F' && number == 0x27 && size == sizeof(nvos02_with_fd)) {
        nvos02_with_fd *p = argument;
        fprintf(stderr,
                "RM_ALLOC_MEMORY before fd=%d root=%08x parent=%08x new=%08x "
                "class=%08x flags=%08x memory=%016llx limit=%llu map_fd=%d\n",
                fd, p->h_root, p->h_parent, p->h_new, p->h_class, p->flags,
                (unsigned long long)p->memory, (unsigned long long)p->limit, p->fd);
    }
    if (type == 'F' && number == 0x2b && size == sizeof(nvos64)) {
        nvos64 *p = argument;
        fprintf(stderr,
                "RM_ALLOC before fd=%d root=%08x parent=%08x new=%08x "
                "class=%08x params=%016llx params_size=%u flags=%08x\n",
                fd, p->h_root, p->h_parent, p->h_new, p->h_class,
                (unsigned long long)p->alloc_params, p->params_size, p->flags);
    }
    if (type == 'F' && number == 0x4e && size == sizeof(nvos33_with_fd)) {
        nvos33_with_fd *p = argument;
        fprintf(stderr,
                "RM_MAP before fd=%d client=%08x device=%08x memory=%08x "
                "offset=%llu length=%llu linear=%016llx flags=%08x map_fd=%d\n",
                fd, p->h_client, p->h_device, p->h_memory,
                (unsigned long long)p->offset, (unsigned long long)p->length,
                (unsigned long long)p->linear_address, p->flags, p->fd);
    }
    if (type == 'F' && number == 0x2a && size == sizeof(nvos54)) {
        nvos54 *p = argument;
        if ((p->command & 0xffff00U) == 0x003d00U) {
            fprintf(stderr,
                    "RM_UNIX_CONTROL before fd=%d client=%08x object=%08x "
                    "cmd=%08x params=%016llx params_size=%u\n",
                    fd, p->h_client, p->h_object, p->command,
                    (unsigned long long)p->params, p->params_size);
        }
        if (p->command == 0x3d0b &&
            p->params_size == sizeof(export_objects_to_fd_params)) {
            export_objects_to_fd_params *e = (void *)(uintptr_t)p->params;
            fprintf(stderr,
                    "RM_EXPORTS before export_fd=%d device=%08x max=%u "
                    "num=%u index=%u first_object=%08x\n",
                    e->fd, e->h_device, e->max_objects, e->num_objects,
                    e->index, e->num_objects ? e->objects[e->index] : 0);
            if (e->num_objects == 1) {
                last_export_client = p->h_client;
                last_export_device = e->h_device;
                last_export_object = e->objects[e->index];
                last_export_control_fd = fd;
            }
        }
        if (p->command == 0x3d05 &&
            p->params_size == sizeof(export_object_to_fd_params)) {
            export_object_to_fd_params *e = (void *)(uintptr_t)p->params;
            fprintf(stderr,
                    "RM_EXPORT before fd=%d client=%08x control_object=%08x "
                    "type=%u device=%08x parent=%08x object=%08x export_fd=%d flags=%08x\n",
                    fd, p->h_client, p->h_object, e->type, e->h_device,
                    e->h_parent, e->h_object, e->fd, e->flags);
        }
    }

    const int result = real_ioctl(fd, request, argument);

    if (type == 'F' && number == 0x27 && size == sizeof(nvos02_with_fd)) {
        nvos02_with_fd *p = argument;
        fprintf(stderr,
                "RM_ALLOC_MEMORY after  rc=%d root=%08x parent=%08x new=%08x "
                "class=%08x flags=%08x memory=%016llx limit=%llu status=%08x map_fd=%d\n",
                result, p->h_root, p->h_parent, p->h_new, p->h_class, p->flags,
                (unsigned long long)p->memory, (unsigned long long)p->limit,
                p->status, p->fd);
    }
    if (type == 'F' && number == 0x2b && size == sizeof(nvos64)) {
        nvos64 *p = argument;
        fprintf(stderr,
                "RM_ALLOC after  rc=%d root=%08x parent=%08x new=%08x "
                "class=%08x params=%016llx params_size=%u flags=%08x status=%08x\n",
                result, p->h_root, p->h_parent, p->h_new, p->h_class,
                (unsigned long long)p->alloc_params, p->params_size, p->flags,
                p->status);
    }
    if (type == 'F' && number == 0x4e && size == sizeof(nvos33_with_fd)) {
        nvos33_with_fd *p = argument;
        fprintf(stderr,
                "RM_MAP after  rc=%d client=%08x device=%08x memory=%08x "
                "offset=%llu length=%llu linear=%016llx status=%08x flags=%08x map_fd=%d\n",
                result, p->h_client, p->h_device, p->h_memory,
                (unsigned long long)p->offset, (unsigned long long)p->length,
                (unsigned long long)p->linear_address, p->status, p->flags,
                p->fd);
    }
    if (type == 'F' && number == 0x2a && size == sizeof(nvos54)) {
        nvos54 *p = argument;
        if ((p->command & 0xffff00U) == 0x003d00U) {
            fprintf(stderr,
                    "RM_UNIX_CONTROL after  rc=%d status=%08x client=%08x "
                    "object=%08x cmd=%08x params_size=%u\n",
                    result, p->status, p->h_client, p->h_object, p->command,
                    p->params_size);
        }
        if (p->command == 0x3d0b &&
            p->params_size == sizeof(export_objects_to_fd_params)) {
            export_objects_to_fd_params *e = (void *)(uintptr_t)p->params;
            fprintf(stderr,
                    "RM_EXPORTS after  export_fd=%d device=%08x max=%u "
                    "num=%u index=%u first_object=%08x\n",
                    e->fd, e->h_device, e->max_objects, e->num_objects,
                    e->index, e->num_objects ? e->objects[e->index] : 0);
        }
        if (p->command == 0x3d05 &&
            p->params_size == sizeof(export_object_to_fd_params)) {
            export_object_to_fd_params *e = (void *)(uintptr_t)p->params;
            fprintf(stderr,
                    "RM_EXPORT after  rc=%d status=%08x client=%08x "
                    "device=%08x parent=%08x object=%08x export_fd=%d flags=%08x\n",
                    result, p->status, p->h_client, e->h_device, e->h_parent,
                    e->h_object, e->fd, e->flags);
        }
    }

    return result;
}
