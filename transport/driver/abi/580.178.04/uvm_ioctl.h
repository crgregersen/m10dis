/*******************************************************************************
    Copyright (c) 2013-2025 NVidia Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
*******************************************************************************/

#ifndef _UVM_IOCTL_H
#define _UVM_IOCTL_H

#include "uvm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// Please see the header file (uvm.h) for detailed documentation on each of the
// associated API calls.
//

#if defined(WIN32) || defined(WIN64)
#   define UVM_IOCTL_BASE(i)       CTL_CODE(FILE_DEVICE_UNKNOWN, 0x800+i, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
#else
#   define UVM_IOCTL_BASE(i) i
#endif

//
// UvmReserveVa
//
#define UVM_RESERVE_VA                                                UVM_IOCTL_BASE(1)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_RESERVE_VA_PARAMS;

//
// UvmReleaseVa
//
#define UVM_RELEASE_VA                                                UVM_IOCTL_BASE(2)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_RELEASE_VA_PARAMS;

//
// UvmRegionCommit
//
#define UVM_REGION_COMMIT                                             UVM_IOCTL_BASE(3)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64           length        NV_ALIGN_BYTES(8); // IN
    UvmStream       streamId      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid gpuUuid;                         // IN
    NV_STATUS       rmStatus;                        // OUT
} UVM_REGION_COMMIT_PARAMS;

//
// UvmRegionDecommit
//
#define UVM_REGION_DECOMMIT                                           UVM_IOCTL_BASE(4)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_REGION_DECOMMIT_PARAMS;

//
// UvmRegionSetStream
//
#define UVM_REGION_SET_STREAM                                         UVM_IOCTL_BASE(5)

typedef struct
{
    NvU64           requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64           length        NV_ALIGN_BYTES(8); // IN
    UvmStream       newStreamId   NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid gpuUuid;                         // IN
    NV_STATUS       rmStatus;                        // OUT
} UVM_REGION_SET_STREAM_PARAMS;

//
// UvmSetStreamRunning
//
#define UVM_SET_STREAM_RUNNING                                        UVM_IOCTL_BASE(6)

typedef struct
{
    UvmStream  streamId NV_ALIGN_BYTES(8);  // IN
    NV_STATUS  rmStatus;                    // OUT
} UVM_SET_STREAM_RUNNING_PARAMS;


//
// Due to limitations in how much we want to send per ioctl call, the nStreams
// member must be less than or equal to about 250. That's an upper limit.
//
// However, from a typical user-space driver's point of view (for example, the
// CUDA driver), a vast majority of the time, we expect there to be only one
// stream passed in. The second most common case is something like atmost 32
// streams being passed in. The cases where there are more than 32 streams are
// the most rare. So we might want to optimize the ioctls accordingly so that we
// don't always copy a 250 * sizeof(streamID) sized array when there's only one
// or a few streams.
//
// For that reason, UVM_MAX_STREAMS_PER_IOCTL_CALL is set to 32.
//
// If the higher-level (uvm.h) call requires more streams to be stopped than
// this value, then multiple ioctl calls should be made.
//
#define UVM_MAX_STREAMS_PER_IOCTL_CALL 32

//
// UvmSetStreamStopped
//
#define UVM_SET_STREAM_STOPPED                                        UVM_IOCTL_BASE(7)

typedef struct
{
    UvmStream streamIdArray[UVM_MAX_STREAMS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     nStreams                                      NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                        // OUT
} UVM_SET_STREAM_STOPPED_PARAMS;

//
// UvmCallTestFunction
//
#define UVM_RUN_TEST                                                  UVM_IOCTL_BASE(9)

typedef struct
{
    NvProcessorUuid gpuUuid;     // IN
    NvU32           test;        // IN
    struct
    {
        NvProcessorUuid peerGpuUuid; // IN
        NvU32           peerId;      // IN
    } multiGpu;
    NV_STATUS      rmStatus;    // OUT
} UVM_RUN_TEST_PARAMS;

//
// This is a magic offset for mmap. Any mapping of an offset above this
// threshold will be treated as a counters mapping, not as an allocation
// mapping. Since allocation offsets must be identical to the virtual address
// of the mapping, this threshold has to be an offset that cannot be
// a valid virtual address.
//
#if defined(__linux__)
    #if defined(NV_64_BITS)
        #define UVM_EVENTS_OFFSET_BASE   (1UL << 63)
        #define UVM_COUNTERS_OFFSET_BASE (1UL << 62)
    #else
        #define UVM_EVENTS_OFFSET_BASE   (1UL << 31)
        #define UVM_COUNTERS_OFFSET_BASE (1UL << 30)
    #endif
#endif // defined(__linux___)

//
// UvmAddSession
//
#define UVM_ADD_SESSION                                               UVM_IOCTL_BASE(10)

typedef struct
{
    NvU32        pidTarget;                             // IN
#ifdef __linux__
    NvP64        countersBaseAddress NV_ALIGN_BYTES(8); // IN
    NvS32        sessionIndex;                          // OUT (session index that got added)
#endif
    NV_STATUS    rmStatus;                              // OUT
} UVM_ADD_SESSION_PARAMS;

//
// UvmRemoveSession
//
#define UVM_REMOVE_SESSION                                             UVM_IOCTL_BASE(11)

typedef struct
{
#ifdef __linux__
    NvS32        sessionIndex; // IN (session index to be removed)
#endif
    NV_STATUS    rmStatus;     // OUT
} UVM_REMOVE_SESSION_PARAMS;


#define UVM_MAX_COUNTERS_PER_IOCTL_CALL 32

//
// UvmEnableCounters
//
#define UVM_ENABLE_COUNTERS                                           UVM_IOCTL_BASE(12)

typedef struct
{
#ifdef __linux__
    NvS32            sessionIndex;                            // IN
#endif
    UvmCounterConfig config[UVM_MAX_COUNTERS_PER_IOCTL_CALL]; // IN
    NvU32            count;                                   // IN
    NV_STATUS        rmStatus;                                // OUT
} UVM_ENABLE_COUNTERS_PARAMS;

//
// UvmMapCounter
//
#define UVM_MAP_COUNTER                                               UVM_IOCTL_BASE(13)

typedef struct
{
#ifdef __linux__
    NvS32           sessionIndex;                   // IN
#endif
    NvU32           scope;                          // IN (UvmCounterScope)
    NvU32           counterName;                    // IN (UvmCounterName)
    NvProcessorUuid gpuUuid;                        // IN
    NvP64           addr         NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                       // OUT
} UVM_MAP_COUNTER_PARAMS;

//
// UvmCreateEventQueue
//
#define UVM_CREATE_EVENT_QUEUE                                        UVM_IOCTL_BASE(14)

typedef struct
{
#ifdef __linux__
    NvS32                 sessionIndex;                         // IN
#endif
    NvU32                 eventQueueIndex;                      // OUT
    NvU64                 queueSize          NV_ALIGN_BYTES(8); // IN
    NvU64                 notificationCount  NV_ALIGN_BYTES(8); // IN
#if defined(WIN32) || defined(WIN64)
    NvU64                 notificationHandle NV_ALIGN_BYTES(8); // IN
#endif
    NvU32                 timeStampType;                        // IN (UvmEventTimeStampType)
    NV_STATUS             rmStatus;                             // OUT
} UVM_CREATE_EVENT_QUEUE_PARAMS;

//
// UvmRemoveEventQueue
//
#define UVM_REMOVE_EVENT_QUEUE                                        UVM_IOCTL_BASE(15)

typedef struct
{
#ifdef __linux__
    NvS32         sessionIndex;       // IN
#endif
    NvU32         eventQueueIndex;    // IN
    NV_STATUS     rmStatus;           // OUT
} UVM_REMOVE_EVENT_QUEUE_PARAMS;

//
// UvmMapEventQueue
//
#define UVM_MAP_EVENT_QUEUE                                           UVM_IOCTL_BASE(16)

typedef struct
{
#ifdef __linux__
    NvS32         sessionIndex;                       // IN
#endif
    NvU32         eventQueueIndex;                    // IN
    NvP64         userRODataAddr   NV_ALIGN_BYTES(8); // IN
    NvP64         userRWDataAddr   NV_ALIGN_BYTES(8); // IN
    NvP64         readIndexAddr    NV_ALIGN_BYTES(8); // OUT
    NvP64         writeIndexAddr   NV_ALIGN_BYTES(8); // OUT
    NvP64         queueBufferAddr  NV_ALIGN_BYTES(8); // OUT
    NV_STATUS     rmStatus;                           // OUT
} UVM_MAP_EVENT_QUEUE_PARAMS;

//
// UvmEnableEvent
//
#define UVM_EVENT_CTRL                                                UVM_IOCTL_BASE(17)

typedef struct
{
#ifdef __linux__
    NvS32        sessionIndex;      // IN
#endif
    NvU32        eventQueueIndex;   // IN
    NvS32        eventType;         // IN
    NvU32        enable;            // IN
    NV_STATUS    rmStatus;          // OUT
} UVM_EVENT_CTRL_PARAMS;

//
// UvmRegisterMpsServer
//
#define UVM_REGISTER_MPS_SERVER                                       UVM_IOCTL_BASE(18)

typedef struct
{
    NvProcessorUuid gpuUuidArray[UVM_MAX_GPUS_V1];                 // IN
    NvU32           numGpus;                                       // IN
    NvU64           serverId                    NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                                      // OUT
} UVM_REGISTER_MPS_SERVER_PARAMS;

//
// UvmRegisterMpsClient
//
#define UVM_REGISTER_MPS_CLIENT                                       UVM_IOCTL_BASE(19)

typedef struct
{
    NvU64     serverId  NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_REGISTER_MPS_CLIENT_PARAMS;

//
// UvmEventGetGpuUuidTable
//
#define UVM_GET_GPU_UUID_TABLE                                        UVM_IOCTL_BASE(20)

typedef struct
{
    NvProcessorUuid gpuUuidArray[UVM_MAX_GPUS_V1]; // OUT
    NvU32           validCount;                    // OUT
    NV_STATUS       rmStatus;                      // OUT
} UVM_GET_GPU_UUID_TABLE_PARAMS;

#if defined(WIN32) || defined(WIN64)
//
// UvmRegionSetBacking
//
#define UVM_REGION_SET_BACKING                                        UVM_IOCTL_BASE(21)

typedef struct
{
    NvProcessorUuid gpuUuid;                        // IN
    NvU32           hAllocation;                    // IN
    NvP64           vaAddr       NV_ALIGN_BYTES(8); // IN
    NvU64           regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                       // OUT
} UVM_REGION_SET_BACKING_PARAMS;

//
// UvmRegionUnsetBacking
//
#define UVM_REGION_UNSET_BACKING                                      UVM_IOCTL_BASE(22)

typedef struct
{
    NvP64     vaAddr       NV_ALIGN_BYTES(8); // IN
    NvU64     regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                       // OUT
} UVM_REGION_UNSET_BACKING_PARAMS;

#endif

#define UVM_CREATE_RANGE_GROUP                                        UVM_IOCTL_BASE(23)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                       // OUT
} UVM_CREATE_RANGE_GROUP_PARAMS;

#define UVM_DESTROY_RANGE_GROUP                                       UVM_IOCTL_BASE(24)

typedef struct
{
    NvU64     rangeGroupId NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                      // OUT
} UVM_DESTROY_RANGE_GROUP_PARAMS;

//
// UvmRegisterGpuVaSpace
//
#define UVM_REGISTER_GPU_VASPACE                                      UVM_IOCTL_BASE(25)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NvS32           rmCtrlFd; // IN
    NvHandle        hClient;  // IN
    NvHandle        hVaSpace; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_REGISTER_GPU_VASPACE_PARAMS;

//
// UvmUnregisterGpuVaSpace
//
#define UVM_UNREGISTER_GPU_VASPACE                                    UVM_IOCTL_BASE(26)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_VASPACE_PARAMS;

//
// UvmRegisterChannel
//
#define UVM_REGISTER_CHANNEL                                          UVM_IOCTL_BASE(27)

typedef struct
{
    NvProcessorUuid gpuUuid;                     // IN
    NvS32           rmCtrlFd;                    // IN
    NvHandle        hClient;                     // IN
    NvHandle        hChannel;                    // IN
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_REGISTER_CHANNEL_PARAMS;

//
// UvmUnregisterChannel
//
#define UVM_UNREGISTER_CHANNEL                                        UVM_IOCTL_BASE(28)

typedef struct
{
    NvProcessorUuid gpuUuid;  // IN
    NvHandle        hClient;  // IN
    NvHandle        hChannel; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_CHANNEL_PARAMS;

//
// UvmEnablePeerAccess
//
#define UVM_ENABLE_PEER_ACCESS                                       UVM_IOCTL_BASE(29)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_ENABLE_PEER_ACCESS_PARAMS;

//
// UvmDisablePeerAccess
//
#define UVM_DISABLE_PEER_ACCESS                                      UVM_IOCTL_BASE(30)

typedef struct
{
    NvProcessorUuid gpuUuidA; // IN
    NvProcessorUuid gpuUuidB; // IN
    NV_STATUS  rmStatus; // OUT
} UVM_DISABLE_PEER_ACCESS_PARAMS;

//
// UvmSetRangeGroup
//
#define UVM_SET_RANGE_GROUP                                           UVM_IOCTL_BASE(31)

typedef struct
{
    NvU64     rangeGroupId  NV_ALIGN_BYTES(8); // IN
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_SET_RANGE_GROUP_PARAMS;

//
// UvmMapExternalAllocation
//
#define UVM_MAP_EXTERNAL_ALLOCATION                                   UVM_IOCTL_BASE(33)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvU64                   offset                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];                    // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NvS32                   rmCtrlFd;                                          // IN
    NvU32                   hClient;                                           // IN
    NvU32                   hMemory;                                           // IN

    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_ALLOCATION_PARAMS;

//
// UvmFree
//
#define UVM_FREE                                                      UVM_IOCTL_BASE(34)
typedef struct
{
    NvU64     base      NV_ALIGN_BYTES(8); // IN
    NvU64     length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                    // OUT
} UVM_FREE_PARAMS;

//
// UvmMemMap
//
#define UVM_MEM_MAP                                                   UVM_IOCTL_BASE(35)

typedef struct
{
    NvP64     regionBase   NV_ALIGN_BYTES(8); // IN
    NvU64     regionLength NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                       // OUT
} UVM_MEM_MAP_PARAMS;

//
// UvmRegisterGpu
//
#define UVM_REGISTER_GPU                                              UVM_IOCTL_BASE(37)

typedef struct
{
    NvProcessorUuid gpu_uuid;    // IN/OUT
    NvBool          numaEnabled; // OUT
    NvS32           numaNodeId;  // OUT
    NvS32           rmCtrlFd;    // IN
    NvHandle        hClient;     // IN
    NvHandle        hSmcPartRef; // IN
    NV_STATUS       rmStatus;    // OUT
} UVM_REGISTER_GPU_PARAMS;

//
// UvmUnregisterGpu
//
#define UVM_UNREGISTER_GPU                                            UVM_IOCTL_BASE(38)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_UNREGISTER_GPU_PARAMS;

#define UVM_PAGEABLE_MEM_ACCESS                                       UVM_IOCTL_BASE(39)

typedef struct
{
    NvBool    pageableMemAccess; // OUT
    NV_STATUS rmStatus;          // OUT
} UVM_PAGEABLE_MEM_ACCESS_PARAMS;

//
// Due to limitations in how much we want to send per ioctl call, the numGroupIds
// member must be less than or equal to about 250. That's an upper limit.
//
// However, from a typical user-space driver's point of view (for example, the
// CUDA driver), a vast majority of the time, we expect there to be only one
// range group passed in. The second most common case is something like atmost 32
// range groups being passed in. The cases where there are more than 32 range
// groups are the most rare. So we might want to optimize the ioctls accordingly
// so that we don't always copy a 250 * sizeof(NvU64) sized array when there's
// only one or a few range groups.
//
// For that reason, UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL is set to 32.
//
// If the higher-level (uvm.h) call requires more range groups than
// this value, then multiple ioctl calls should be made.
//
#define UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL 32

//
// UvmPreventMigrationRangeGroups
//
#define UVM_PREVENT_MIGRATION_RANGE_GROUPS                            UVM_IOCTL_BASE(40)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                             // OUT
} UVM_PREVENT_MIGRATION_RANGE_GROUPS_PARAMS;

//
// UvmAllowMigrationRangeGroups
//
#define UVM_ALLOW_MIGRATION_RANGE_GROUPS                              UVM_IOCTL_BASE(41)

typedef struct
{
    NvU64     rangeGroupIds[UVM_MAX_RANGE_GROUPS_PER_IOCTL_CALL] NV_ALIGN_BYTES(8); // IN
    NvU64     numGroupIds                                        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                                             // OUT
} UVM_ALLOW_MIGRATION_RANGE_GROUPS_PARAMS;

//
// UvmSetPreferredLocation
//
#define UVM_SET_PREFERRED_LOCATION                                    UVM_IOCTL_BASE(42)

typedef struct
{
    NvU64           requestedBase      NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid preferredLocation;                    // IN
    NvS32           preferredCpuNumaNode;                 // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_SET_PREFERRED_LOCATION_PARAMS;

//
// UvmUnsetPreferredLocation
//
#define UVM_UNSET_PREFERRED_LOCATION                                  UVM_IOCTL_BASE(43)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_UNSET_PREFERRED_LOCATION_PARAMS;

//
// UvmEnableReadDuplication
//
#define UVM_ENABLE_READ_DUPLICATION                                   UVM_IOCTL_BASE(44)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_ENABLE_READ_DUPLICATION_PARAMS;

//
// UvmDisableReadDuplication
//
#define UVM_DISABLE_READ_DUPLICATION                                  UVM_IOCTL_BASE(45)

typedef struct
{
    NvU64     requestedBase NV_ALIGN_BYTES(8); // IN
    NvU64     length        NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                        // OUT
} UVM_DISABLE_READ_DUPLICATION_PARAMS;

//
// UvmSetAccessedBy
//
#define UVM_SET_ACCESSED_BY                                           UVM_IOCTL_BASE(46)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
    NvU64           length          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid accessedByUuid;                    // IN
    NV_STATUS       rmStatus;                          // OUT
} UVM_SET_ACCESSED_BY_PARAMS;

//
// UvmUnsetAccessedBy
//
#define UVM_UNSET_ACCESSED_BY                                         UVM_IOCTL_BASE(47)

typedef struct
{
    NvU64           requestedBase   NV_ALIGN_BYTES(8); // IN
    NvU64           length          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid accessedByUuid;                    // IN
    NV_STATUS       rmStatus;                          // OUT
} UVM_UNSET_ACCESSED_BY_PARAMS;

// For managed allocations, UVM_MIGRATE implements the behavior described in
// UvmMigrate. If the input virtual range corresponds to system-allocated
// pageable memory, and the GPUs in the system support transparent access to
// pageable memory, the scheme is a bit more elaborate, potentially with
// several transitions betwen user and kernel spaces:
//
// 1) UVM_MIGRATE with the range base address and size. This will migrate
// anonymous vmas until:
//   a) It finds a file-backed vma or no GPUs are registered in the VA space
//      so no GPU can drive the copy. It will try to populate the vma using
//      get_user_pages and return NV_WARN_NOTHING_TO_DO.
//   b) It fails to allocate memory on the destination CPU node. It will return
//       NV_ERR_MORE_PROCESSING_REQUIRED.
//   c) It fails to populate pages directly on the destination GPU. It will try
//      to populate the vma using get_user_pages and return.
//   d) The full input range is migrated (or empty), this call will release
//      the semaphore before returning.
// 2) The user-mode needs to handle the following error codes:
//   a) NV_WARN_NOTHING_TO_DO: use move_pages to migrate pages for the VA
//      range corresponding to the vma that couldn't be migrated in kernel
//      mode. Then, it processes the remainder of the range, starting after
//      that vma.
//   b) NV_ERR_MORE_PROCESSING_REQUIRED: choose a different CPU NUMA node,
//      trying to enforce the NUMA policies of the thread and retry the
//      ioctl. If there are no more CPU NUMA nodes to try, try to populate
//      the remainder of the range anywhere using the UVM_POPULATE_PAGEABLE
//      ioctl.
//   c) NV_OK: success. This only guarantees that pages were populated, not
//      that they moved to the requested destination.
// 3) For cases 2.a) and 2.b) Goto 1
//
// If UVM_MIGRATE_FLAG_ASYNC is 0, the ioctl won't return until the migration is
// done and all mappings are updated, subject to the special rules for pageable
// memory described above. semaphoreAddress must be 0. semaphorePayload is
// ignored.
//
// If UVM_MIGRATE_FLAG_ASYNC is 1, the ioctl may return before the migration is
// complete. If semaphoreAddress is 0, semaphorePayload is ignored and no
// notification will be given on completion. If semaphoreAddress is non-zero
// and the returned error code is NV_OK, semaphorePayload will be written to
// semaphoreAddress once the migration is complete.
#define UVM_MIGRATE_FLAG_ASYNC              0x00000001

// When the migration destination is the CPU, skip the step which creates new
// virtual mappings on the CPU. Creating CPU mappings must wait for the
// migration to complete, so skipping this step allows the migration to be
// fully asynchronous. This flag is ignored for pageable migrations if the GPUs
// in the system support transparent access to pageable memory.
//
// The UVM driver must have builtin tests enabled for the API to use this flag.
#define UVM_MIGRATE_FLAG_SKIP_CPU_MAP       0x00000002

// By default UVM_MIGRATE returns an error if the destination UUID is a GPU
// without a registered GPU VA space. Setting this flag skips that check, so the
// destination GPU only needs to have been registered.
//
// This can be used in tests to trigger migrations of physical memory without
// the overhead of GPU PTE mappings.
//
// The UVM driver must have builtin tests enabled for the API to use this flag.
#define UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE    0x00000004

#define UVM_MIGRATE_FLAGS_TEST_ALL              (UVM_MIGRATE_FLAG_SKIP_CPU_MAP      | \
                                                 UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE)

#define UVM_MIGRATE_FLAGS_ALL                   (UVM_MIGRATE_FLAG_ASYNC | \
                                                 UVM_MIGRATE_FLAGS_TEST_ALL)

// If NV_ERR_INVALID_ARGUMENT is returned it is because cpuMemoryNode is not
// valid and the destination processor is the CPU. cpuMemoryNode is considered
// invalid if:
//      * it is less than -1,
//      * it is equal to or larger than the maximum number of nodes, or
//      * it corresponds to a registered GPU.
//      * it is not in the node_possible_map set of nodes,
//      * it does not have onlined memory
//
// For pageable migrations:
//
// In addition to the above, in the case of pageable memory, the
// cpuMemoryNode is considered invalid if it's -1.
//
// If NV_WARN_NOTHING_TO_DO is returned, user-space is responsible for
// completing the migration of the VA range described by userSpaceStart and
// userSpaceLength using move_pages.
//
// If NV_ERR_MORE_PROCESSING_REQUIRED is returned, user-space is responsible
// for re-trying with a different cpuNumaNode, starting at userSpaceStart.
//
#define UVM_MIGRATE                                                   UVM_IOCTL_BASE(51)
typedef struct
{
    NvU64           base               NV_ALIGN_BYTES(8); // IN
    NvU64           length             NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NvU32           flags;                                // IN
    NvU64           semaphoreAddress   NV_ALIGN_BYTES(8); // IN
    NvU32           semaphorePayload;                     // IN
    NvS32           cpuNumaNode;                          // IN
    NvU64           userSpaceStart     NV_ALIGN_BYTES(8); // OUT
    NvU64           userSpaceLength    NV_ALIGN_BYTES(8); // OUT
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_PARAMS;

#define UVM_MIGRATE_RANGE_GROUP                                       UVM_IOCTL_BASE(53)
typedef struct
{
    NvU64           rangeGroupId       NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid destinationUuid;                      // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_MIGRATE_RANGE_GROUP_PARAMS;

//
// UvmEnableSystemWideAtomics
//
#define UVM_ENABLE_SYSTEM_WIDE_ATOMICS                                UVM_IOCTL_BASE(54)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_ENABLE_SYSTEM_WIDE_ATOMICS_PARAMS;

//
// UvmDisableSystemWideAtomics
//
#define UVM_DISABLE_SYSTEM_WIDE_ATOMICS                               UVM_IOCTL_BASE(55)

typedef struct
{
    NvProcessorUuid gpu_uuid; // IN
    NV_STATUS       rmStatus; // OUT
} UVM_DISABLE_SYSTEM_WIDE_ATOMICS_PARAMS;

//
// Initialize any tracker object such as a queue or counter
// UvmToolsCreateEventQueue, UvmToolsCreateProcessAggregateCounters,
// UvmToolsCreateProcessorCounters.
//
#define UVM_TOOLS_INIT_EVENT_TRACKER                                  UVM_IOCTL_BASE(56)
typedef struct
{
    NvU64           queueBuffer        NV_ALIGN_BYTES(8); // IN
    NvU64           queueBufferSize    NV_ALIGN_BYTES(8); // IN
    NvU64           controlBuffer      NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid processor;                            // IN
    NvU32           allProcessors;                        // IN
    NvU32           uvmFd;                                // IN
    NV_STATUS       rmStatus;                             // OUT
} UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS;

//
// UvmToolsSetNotificationThreshold
//
#define UVM_TOOLS_SET_NOTIFICATION_THRESHOLD                          UVM_IOCTL_BASE(57)
typedef struct
{
    NvU32     notificationThreshold;                       // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_SET_NOTIFICATION_THRESHOLD_PARAMS;

//
// UvmToolsEventQueueEnableEvents
//
#define UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS                           UVM_IOCTL_BASE(58)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_EVENT_QUEUE_ENABLE_EVENTS_PARAMS;

//
// UvmToolsEventQueueDisableEvents
//
#define UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS                          UVM_IOCTL_BASE(59)
typedef struct
{
    NvU64     eventTypeFlags            NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_EVENT_QUEUE_DISABLE_EVENTS_PARAMS;

//
// UvmToolsEnableCounters
//
#define UVM_TOOLS_ENABLE_COUNTERS                                     UVM_IOCTL_BASE(60)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_ENABLE_COUNTERS_PARAMS;

//
// UvmToolsDisableCounters
//
#define UVM_TOOLS_DISABLE_COUNTERS                                    UVM_IOCTL_BASE(61)
typedef struct
{
    NvU64     counterTypeFlags          NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_DISABLE_COUNTERS_PARAMS;

//
// UvmToolsReadProcessMemory
//
#define UVM_TOOLS_READ_PROCESS_MEMORY                                 UVM_IOCTL_BASE(62)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8); // IN
    NvU64     size                      NV_ALIGN_BYTES(8); // IN
    NvU64     targetVa                  NV_ALIGN_BYTES(8); // IN
    NvU64     bytesRead                 NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_READ_PROCESS_MEMORY_PARAMS;

//
// UvmToolsWriteProcessMemory
//
#define UVM_TOOLS_WRITE_PROCESS_MEMORY                                UVM_IOCTL_BASE(63)
typedef struct
{
    NvU64     buffer                    NV_ALIGN_BYTES(8); // IN
    NvU64     size                      NV_ALIGN_BYTES(8); // IN
    NvU64     targetVa                  NV_ALIGN_BYTES(8); // IN
    NvU64     bytesWritten              NV_ALIGN_BYTES(8); // OUT
    NV_STATUS rmStatus;                                    // OUT
} UVM_TOOLS_WRITE_PROCESS_MEMORY_PARAMS;

//
// UvmToolsGetProcessorUuidTable
//
#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE                            UVM_IOCTL_BASE(64)
typedef struct
{
    NvU64     tablePtr                 NV_ALIGN_BYTES(8); // IN
    NV_STATUS rmStatus;                                   // OUT
} UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS;

//
// UvmMapDynamicParallelismRegion
//
#define UVM_MAP_DYNAMIC_PARALLELISM_REGION                            UVM_IOCTL_BASE(65)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_DYNAMIC_PARALLELISM_REGION_PARAMS;

//
// UvmUnmapExternal
//
#define UVM_UNMAP_EXTERNAL                                            UVM_IOCTL_BASE(66)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_UNMAP_EXTERNAL_PARAMS;


//
// UvmToolsFlushEvents
//
#define UVM_TOOLS_FLUSH_EVENTS                                        UVM_IOCTL_BASE(67)
typedef struct
{
    NV_STATUS rmStatus;                                   // OUT
} UVM_TOOLS_FLUSH_EVENTS_PARAMS;

//
// UvmAllocSemaphorePool
//
#define UVM_ALLOC_SEMAPHORE_POOL                                     UVM_IOCTL_BASE(68)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    UvmGpuMappingAttributes perGpuAttributes[UVM_MAX_GPUS];                    // IN
    NvU64                   gpuAttributesCount              NV_ALIGN_BYTES(8); // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_ALLOC_SEMAPHORE_POOL_PARAMS;

//
// UvmCleanUpZombieResources
//
#define UVM_CLEAN_UP_ZOMBIE_RESOURCES                                 UVM_IOCTL_BASE(69)
typedef struct
{
    NV_STATUS rmStatus;                    // OUT
} UVM_CLEAN_UP_ZOMBIE_RESOURCES_PARAMS;

//
// UvmIsPageableMemoryAccessSupportedOnGpu
//
#define UVM_PAGEABLE_MEM_ACCESS_ON_GPU                                UVM_IOCTL_BASE(70)

typedef struct
{
    NvProcessorUuid gpu_uuid;          // IN
    NvBool          pageableMemAccess; // OUT
    NV_STATUS       rmStatus;          // OUT
} UVM_PAGEABLE_MEM_ACCESS_ON_GPU_PARAMS;

//
// UvmPopulatePageable
//
#define UVM_POPULATE_PAGEABLE                                         UVM_IOCTL_BASE(71)

// Allow population of managed ranges. The goal is to validate that it is
// possible to populate pageable ranges backed by VMAs with the VM_MIXEDMAP or
// VM_DONTEXPAND special flags set. But since there is no portable way to force
// allocation of such memory from user space, and it is not safe to change the
// flags of an already-created VMA from kernel space, we take advantage of the
// fact that managed ranges have both special flags set at creation time (see
// uvm_mmap).
#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED              0x00000001

// By default UVM_POPULATE_PAGEABLE returns an error if the destination vma
// does not have read permission. This flag skips that check.
#define UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK            0x00000002

// By default UVM_POPULATE_PAGEABLE returns an error if the destination vma
// is VM_IO or VM_PFNMAP. This flag skips that check.
#define UVM_POPULATE_PAGEABLE_FLAG_ALLOW_SPECIAL              0x00000004

// These flags are used internally within the driver and are not allowed from
// user space.
#define UVM_POPULATE_PAGEABLE_FLAGS_INTERNAL    UVM_POPULATE_PAGEABLE_FLAG_ALLOW_SPECIAL

// These flags are allowed from user space only when builtin tests are enabled.
// Some of them may also be used internally within the driver in non-test use
// cases.
#define UVM_POPULATE_PAGEABLE_FLAGS_TEST        (UVM_POPULATE_PAGEABLE_FLAG_ALLOW_MANAGED | \
                                                 UVM_POPULATE_PAGEABLE_FLAG_SKIP_PROT_CHECK)

#define UVM_POPULATE_PAGEABLE_FLAGS_ALL         (UVM_POPULATE_PAGEABLE_FLAGS_INTERNAL | \
                                                 UVM_POPULATE_PAGEABLE_FLAGS_TEST)

typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NvU32           flags;                       // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_POPULATE_PAGEABLE_PARAMS;

//
// UvmValidateVaRange
//
#define UVM_VALIDATE_VA_RANGE                                         UVM_IOCTL_BASE(72)
typedef struct
{
    NvU64           base      NV_ALIGN_BYTES(8); // IN
    NvU64           length    NV_ALIGN_BYTES(8); // IN
    NV_STATUS       rmStatus;                    // OUT
} UVM_VALIDATE_VA_RANGE_PARAMS;

#define UVM_CREATE_EXTERNAL_RANGE                                     UVM_IOCTL_BASE(73)
typedef struct
{
    NvU64                  base                             NV_ALIGN_BYTES(8); // IN
    NvU64                  length                           NV_ALIGN_BYTES(8); // IN
    NV_STATUS              rmStatus;                                           // OUT
} UVM_CREATE_EXTERNAL_RANGE_PARAMS;

#define UVM_MAP_EXTERNAL_SPARSE                                       UVM_IOCTL_BASE(74)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NV_STATUS               rmStatus;                                          // OUT
} UVM_MAP_EXTERNAL_SPARSE_PARAMS;

//
// Used to initialise a secondary UVM file-descriptor which holds a
// reference on the memory map to prevent it being torn down without
// first notifying UVM. This is achieved by preventing mmap() calls on
// the secondary file-descriptor so that on process exit
// uvm_mm_release() will be called while the memory map is present
// such that UVM can cleanly shutdown the GPU by handling faults
// instead of cancelling them.
//
// This ioctl must be called after the primary file-descriptor has
// been initialised with the UVM_INITIALIZE ioctl. The primary FD
// should be passed in the uvmFd field and the UVM_MM_INITIALIZE ioctl
// will hold a reference on the primary FD. Therefore uvm_release() is
// guaranteed to be called after uvm_mm_release().
//
// Once this file-descriptor has been closed the UVM context is
// effectively dead and subsequent operations requiring a memory map
// will fail. Calling UVM_MM_INITIALIZE on a context that has already
// been initialized via any FD will return NV_ERR_INVALID_STATE.
//
// Calling this with a non-UVM file-descriptor in uvmFd will return
// NV_ERR_INVALID_ARGUMENT. Calling this on the same file-descriptor
// as UVM_INITIALIZE or more than once on the same FD will return
// NV_ERR_IN_USE.
//
// Not all platforms require this secondary file-descriptor. On those
// platforms NV_WARN_NOTHING_TO_DO will be returned and users may
// close the file-descriptor at anytime.
#define UVM_MM_INITIALIZE                                             UVM_IOCTL_BASE(75)
typedef struct
{
    NvS32                   uvmFd;    // IN
    NV_STATUS               rmStatus; // OUT
} UVM_MM_INITIALIZE_PARAMS;

#define UVM_TOOLS_INIT_EVENT_TRACKER_V2                               UVM_IOCTL_BASE(76)
typedef UVM_TOOLS_INIT_EVENT_TRACKER_PARAMS UVM_TOOLS_INIT_EVENT_TRACKER_V2_PARAMS;

#define UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2                         UVM_IOCTL_BASE(77)
typedef UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_PARAMS UVM_TOOLS_GET_PROCESSOR_UUID_TABLE_V2_PARAMS;

//
// UvmAllocDeviceP2P
//
#define UVM_ALLOC_DEVICE_P2P                                          UVM_IOCTL_BASE(78)
typedef struct
{
    NvU64                   base                            NV_ALIGN_BYTES(8); // IN
    NvU64                   length                          NV_ALIGN_BYTES(8); // IN
    NvU64                   offset                          NV_ALIGN_BYTES(8); // IN
    NvProcessorUuid         gpuUuid;                                           // IN
    NvS32                   rmCtrlFd;                                          // IN
    NvU32                   hClient;                                           // IN
    NvU32                   hMemory;                                           // IN

    NV_STATUS               rmStatus;                                          // OUT
} UVM_ALLOC_DEVICE_P2P_PARAMS;

#define UVM_CLEAR_ALL_ACCESS_COUNTERS                                 UVM_IOCTL_BASE(79)

typedef struct
{
    NV_STATUS       rmStatus; // OUT
} UVM_CLEAR_ALL_ACCESS_COUNTERS_PARAMS;

//
// UvmDiscard
//
#define UVM_DISCARD                                                   UVM_IOCTL_BASE(80)
typedef struct
{
    NvU64           base                                    NV_ALIGN_BYTES(8); // IN
    NvU64           length;                                                    // IN
    NvU64           flags;                                                     // IN
    NV_STATUS       rmStatus;                                                  // OUT
} UVM_DISCARD_PARAMS;

// GLM53 read-only local vidmem physical-offset probe.
//
#define UVM_GLM53_READ_PHYS_VIDMEM                                    UVM_IOCTL_BASE(81)
typedef struct
{
    NvProcessorUuid gpu_uuid;                                             // IN
    NvU64           phys_offset                       NV_ALIGN_BYTES(8);  // IN: local VIDMEM physical offset
    NvU64           length                            NV_ALIGN_BYTES(8);  // IN
    NvU64           user_out                          NV_ALIGN_BYTES(8);  // IN: user buffer pointer
    NvU64           bytes_read                        NV_ALIGN_BYTES(8);  // OUT
    NvU64           fb_size                           NV_ALIGN_BYTES(8);  // OUT
    NvU64           fb_phys_start                     NV_ALIGN_BYTES(8);  // OUT
    NvU64           fb_max_allocatable_address        NV_ALIGN_BYTES(8);  // OUT
    NV_STATUS       rmStatus;                                             // OUT
} UVM_GLM53_READ_PHYS_VIDMEM_PARAMS;

//
// GLM53 allocation-backed self-test for UVM_GLM53_READ_PHYS_VIDMEM.
//
#define UVM_GLM53_PHYS_READER_SELF_TEST                               UVM_IOCTL_BASE(82)
typedef struct
{
    NvProcessorUuid gpu_uuid;                                             // IN
    NvU64           length                            NV_ALIGN_BYTES(8);  // IN: 0 selects default
    NvU64           seed                              NV_ALIGN_BYTES(8);  // IN: 0 selects default
    NvU64           bytes_checked                     NV_ALIGN_BYTES(8);  // OUT
    NvU64           vidmem_phys_offset                NV_ALIGN_BYTES(8);  // OUT
    NvU64           vidmem_gpu_va                     NV_ALIGN_BYTES(8);  // OUT: UVM internal VA used to fill
    NvU64           fb_size                           NV_ALIGN_BYTES(8);  // OUT
    NvU64           fb_phys_start                     NV_ALIGN_BYTES(8);  // OUT
    NvU64           fb_max_allocatable_address        NV_ALIGN_BYTES(8);  // OUT
    NvU32           pass_count;                                           // OUT
    NvU32           mismatch_count;                                       // OUT
    NvU32           first_word_pass1;                                     // OUT
    NvU32           first_word_pass2;                                     // OUT
    NV_STATUS       rmStatus;                                             // OUT
} UVM_GLM53_PHYS_READER_SELF_TEST_PARAMS;

//
// GLM53 read-only UVM VA-space PDB metadata snapshot.
//
#define UVM_GLM53_LIST_VA_SPACE_PDBS                                  UVM_IOCTL_BASE(83)
typedef struct
{
    NvProcessorUuid gpu_uuid;
    NvU64           va_space_ptr                     NV_ALIGN_BYTES(8);
    NvU64           gpu_va_space_ptr                 NV_ALIGN_BYTES(8);
    NvU64           mm_ptr                           NV_ALIGN_BYTES(8);
    NvU64           pdb_address                      NV_ALIGN_BYTES(8);
    NvU64           root_phys_address                NV_ALIGN_BYTES(8);
    NvU64           pdb_rm_dma_address               NV_ALIGN_BYTES(8);
    NvU64           big_page_size                    NV_ALIGN_BYTES(8);
    NvU64           range_count                      NV_ALIGN_BYTES(8);
    NvU32           current_pid;
    NvU32           target_pid;
    NvU32           gpu_id_value;
    NvU32           gpu_index;
    NvU32           state;
    NvU32           did_set_page_directory;
    NvU32           pdb_aperture;
    NvU32           root_phys_aperture;
    NvU32           pdb_rm_dma_aperture;
} UVM_GLM53_VA_SPACE_PDB_ENTRY;

typedef struct
{
    NvU32           target_pid;                                           // IN: 0 means current process
    NvU32           entry_capacity;                                       // IN
    NvU64           entries                           NV_ALIGN_BYTES(8);  // IN: user pointer to entries
    NvU32           entry_count;                                          // OUT
    NvU32           matched_va_spaces;                                    // OUT
    NvU32           total_va_spaces;                                      // OUT
    NvU32           overflow;                                             // OUT
    NV_STATUS       rmStatus;                                             // OUT
} UVM_GLM53_LIST_VA_SPACE_PDBS_PARAMS;

//
// GLM53 restricted local VIDMEM PTE remap transaction.
//
#define UVM_GLM53_LOCAL_PTE_REMAP                                     UVM_IOCTL_BASE(84)
#define UVM_GLM53_LOCAL_PTE_REMAP_OP_APPLY                            1U
#define UVM_GLM53_LOCAL_PTE_REMAP_OP_RESTORE                          2U
#define UVM_GLM53_LOCAL_PTE_REMAP_OP_QUERY                            3U
#define UVM_GLM53_LOCAL_PTE_REMAP_OP_APPLY_VOLATILE                   4U

typedef struct
{
    NvProcessorUuid gpu_uuid;                                             // IN
    NvU32           op;                                                   // IN
    NvU32           root_phys_aperture;                                   // IN
    NvU64           va_space_ptr                     NV_ALIGN_BYTES(8);   // IN
    NvU64           gpu_va_space_ptr                 NV_ALIGN_BYTES(8);   // IN
    NvU64           root_phys_address                NV_ALIGN_BYTES(8);   // IN
    NvU64           target_va                        NV_ALIGN_BYTES(8);   // IN
    NvU64           page_size                        NV_ALIGN_BYTES(8);   // IN
    NvU64           target_pte_phys                  NV_ALIGN_BYTES(8);   // IN
    NvU64           expected_current_pte             NV_ALIGN_BYTES(8);   // IN
    NvU64           replacement_source_pte           NV_ALIGN_BYTES(8);   // IN
    NvU64           pte_before                       NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_written                      NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_after                        NV_ALIGN_BYTES(8);   // OUT
    NvU64           saved_original_pte               NV_ALIGN_BYTES(8);   // OUT
    NvU64           temporary_pte                    NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_pte_phys           NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_va                 NV_ALIGN_BYTES(8);   // OUT
    NvU32           active;                                               // OUT
    NvU32           retained_gpu_va_space_state;                          // OUT
    NV_STATUS       rmStatus;                                             // OUT
} UVM_GLM53_LOCAL_PTE_REMAP_PARAMS;

//
// GLM53 same-M10 peer destination route probe.
//
#define UVM_GLM53_PEER_ROUTE                                          UVM_IOCTL_BASE(85)
#define UVM_GLM53_PEER_ROUTE_OP_QUERY                                 1U
#define UVM_GLM53_PEER_ROUTE_OP_NATIVE_PROBE                          2U
#define UVM_GLM53_PEER_ROUTE_OP_BAR1_SETUP                            3U
#define UVM_GLM53_PEER_ROUTE_OP_CACHE_SYNC                            4U
#define UVM_GLM53_PEER_ROUTE_OP_RELEASE                               5U

#define UVM_GLM53_PEER_ROUTE_TYPE_NONE                                0U
#define UVM_GLM53_PEER_ROUTE_TYPE_NATIVE                              1U
#define UVM_GLM53_PEER_ROUTE_TYPE_BAR1_SYS                            2U

typedef struct
{
    NvProcessorUuid requester_uuid;                                      // IN
    NvProcessorUuid owner_uuid;                                          // IN
    NvU32           op;                                                  // IN
    NvU32           requested_route_type;                                // IN
    NvU64           owner_va                         NV_ALIGN_BYTES(8);  // IN: source VA for BAR1 setup
    NvU64           length                           NV_ALIGN_BYTES(8);  // IN
    NvU64           route_id                         NV_ALIGN_BYTES(8);  // IN/OUT

    NvU32           active;                                              // OUT
    NvU32           active_route_type;                                   // OUT
    NvU32           native_retained;                                     // OUT
    NvU32           native_link_type;                                    // OUT
    NvU32           native_peer_aperture;                                // OUT
    NvU32           native_peer_id;                                      // OUT
    NV_STATUS       native_status;                                       // OUT

    NvS32           p2p_get_pages_rc;                                    // OUT
    NvS32           p2p_dma_map_rc;                                      // OUT
    NvS32           p2p_dma_unmap_rc;                                    // OUT
    NvS32           p2p_put_pages_rc;                                    // OUT
    NvU32           p2p_page_size_type;                                  // OUT
    NvU32           p2p_entries;                                         // OUT
    NvU32           p2p_page_table_flags;                                // OUT
    NvU32           dma_contiguous_for_length;                           // OUT
    NvU32           dma_aligned_for_128k_leaf;                           // OUT
    NvU32           dma_in_requester_window;                             // OUT

    NvU64           p2p_page_size_bytes              NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page0_physical               NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page1_physical               NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page0_dma                    NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page1_dma                    NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page0_gpu_sys                NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page1_gpu_sys                NV_ALIGN_BYTES(8);  // OUT
    NvU64           candidate_gpu_sys_address        NV_ALIGN_BYTES(8);  // OUT
    NvU64           validated_length                 NV_ALIGN_BYTES(8);  // OUT

    NvU64           requester_dma_addressable_start  NV_ALIGN_BYTES(8);  // OUT
    NvU64           requester_dma_addressable_limit  NV_ALIGN_BYTES(8);  // OUT
    NvU64           owner_static_bar1_start          NV_ALIGN_BYTES(8);  // OUT
    NvU64           owner_static_bar1_size           NV_ALIGN_BYTES(8);  // OUT

    NV_STATUS       requester_l2_invalidate_status;                      // OUT
    NV_STATUS       requester_phys_tlb_invalidate_status;                 // OUT
    NV_STATUS       rmStatus;                                            // OUT
} UVM_GLM53_PEER_ROUTE_PARAMS;

//
// GLM53 restricted requester PTE remap to an established peer destination.
//
#define UVM_GLM53_PEER_READ_REMAP                                    UVM_IOCTL_BASE(86)
#define UVM_GLM53_PEER_READ_REMAP_OP_APPLY                           1U
#define UVM_GLM53_PEER_READ_REMAP_OP_RESTORE                         2U
#define UVM_GLM53_PEER_READ_REMAP_OP_QUERY                           3U

typedef struct
{
    NvProcessorUuid requester_uuid;                                      // IN
    NvProcessorUuid owner_uuid;                                          // IN
    NvU32           op;                                                  // IN
    NvU32           route_type;                                          // IN
    NvU32           root_phys_aperture;                                  // IN
    NvU32           reserved0;                                           // IN
    NvU64           route_id                         NV_ALIGN_BYTES(8);  // IN
    NvU64           va_space_ptr                    NV_ALIGN_BYTES(8);   // IN
    NvU64           gpu_va_space_ptr                NV_ALIGN_BYTES(8);   // IN
    NvU64           root_phys_address               NV_ALIGN_BYTES(8);   // IN
    NvU64           target_va                       NV_ALIGN_BYTES(8);   // IN
    NvU64           page_size                       NV_ALIGN_BYTES(8);   // IN
    NvU64           target_pte_phys                 NV_ALIGN_BYTES(8);   // IN
    NvU64           expected_current_pte            NV_ALIGN_BYTES(8);   // IN

    NvU64           pte_before                      NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_written                     NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_after                       NV_ALIGN_BYTES(8);   // OUT
    NvU64           saved_original_pte              NV_ALIGN_BYTES(8);   // OUT
    NvU64           temporary_pte                   NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_pte_phys          NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_va                NV_ALIGN_BYTES(8);   // OUT
    NvU64           destination_gpu_sys_address     NV_ALIGN_BYTES(8);   // OUT
    NvU64           decoded_sys_address             NV_ALIGN_BYTES(8);   // OUT

    NvU32           active;                                              // OUT
    NvU32           retained_gpu_va_space_state;                         // OUT
    NvU32           decoded_aperture_bits;                               // OUT
    NvU32           decoded_kind;                                        // OUT
    NvU32           decoded_read_only;                                   // OUT
    NvU32           decoded_read_disable;                                // OUT
    NvU32           decoded_write_disable;                               // OUT
    NvU32           decoded_volatile;                                    // OUT
    NV_STATUS       rmStatus;                                            // OUT
} UVM_GLM53_PEER_READ_REMAP_PARAMS;

//
// GLM53 one-GPU owner-page export diagnosis.
//
#define UVM_GLM53_OWNER_EXPORT                                        UVM_IOCTL_BASE(87)
#define UVM_GLM53_OWNER_EXPORT_OP_QUERY                              1U
#define UVM_GLM53_OWNER_EXPORT_OP_ACQUIRE_PERSISTENT                 2U
#define UVM_GLM53_OWNER_EXPORT_OP_RELEASE                            3U
#define UVM_GLM53_OWNER_EXPORT_OP_FAIL_BEFORE_ACQUIRE_SELF_TEST      4U

typedef struct
{
    NvProcessorUuid owner_uuid;                                         // IN
    NvU32           op;                                                 // IN
    NvU32           p2p_flags;                                          // IN
    NvU64           owner_va                        NV_ALIGN_BYTES(8);  // IN
    NvU64           length                          NV_ALIGN_BYTES(8);  // IN
    NvU64           export_id                       NV_ALIGN_BYTES(8);  // IN/OUT

    NvU32           active;                                             // OUT
    NvU32           page_table_owned;                                   // OUT
    NvU32           failed_pointer_nonnull;                             // OUT
    NvU32           returned_uuid_matches_owner;                        // OUT
    NvS32           p2p_get_pages_rc;                                   // OUT
    NvS32           p2p_put_pages_rc;                                   // OUT

    NvU32           p2p_page_table_version;                             // OUT
    NvU32           p2p_page_size_type;                                 // OUT
    NvU32           p2p_entries;                                        // OUT
    NvU32           p2p_page_table_flags;                               // OUT
    NvU64           p2p_page_size_bytes             NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page0_physical              NV_ALIGN_BYTES(8);  // OUT
    NvU64           p2p_page1_physical              NV_ALIGN_BYTES(8);  // OUT
    NvU64           active_owner_va                 NV_ALIGN_BYTES(8);  // OUT
    NvU64           active_length                   NV_ALIGN_BYTES(8);  // OUT
    NvU8            returned_gpu_uuid[16];                              // OUT
    NV_STATUS       rmStatus;                                           // OUT
} UVM_GLM53_OWNER_EXPORT_PARAMS;

//
// GLM53 RM-managed CPU mapping probe for test-owned RM allocations.
//
#define UVM_GLM53_RM_CPU_MAP_PROBE                                    UVM_IOCTL_BASE(88)
#define UVM_GLM53_RM_CPU_MAP_PROBE_OP_QUERY                           1U
#define UVM_GLM53_RM_CPU_MAP_PROBE_OP_ACQUIRE                         2U
#define UVM_GLM53_RM_CPU_MAP_PROBE_OP_RELEASE                         3U

#define UVM_GLM53_RM_CPU_MAP_PROBE_TYPE_GPU                           1U
#define UVM_GLM53_RM_CPU_MAP_PROBE_TYPE_SYS                           2U

typedef struct
{
    NvProcessorUuid gpu_uuid;                                           // IN
    NvU32           op;                                                 // IN
    NvU32           mem_type;                                           // IN
    NvU64           length                         NV_ALIGN_BYTES(8);   // IN
    NvU64           gpu_alignment                  NV_ALIGN_BYTES(8);   // IN
    NvU64           seed                           NV_ALIGN_BYTES(8);   // IN
    NvU64           mapping_id                     NV_ALIGN_BYTES(8);   // IN/OUT

    NvU32           active;                                             // OUT
    NvU32           active_mem_type;                                    // OUT
    NvU32           mismatch_count;                                     // OUT
    NvU32           words_checked;                                      // OUT
    NvU32           first_word_expected;                                // OUT
    NvU32           first_word_observed;                                // OUT
    NvU32           second_epoch_word_expected;                         // OUT
    NvU32           second_epoch_word_observed;                         // OUT
    NvU32           free_completed;                                     // OUT
    NvU32           reserved0;                                          // OUT

    NvU64           active_length                  NV_ALIGN_BYTES(8);   // OUT
    NvU64           gpu_va                         NV_ALIGN_BYTES(8);   // OUT: UVM internal GPU VA
    NvU64           cpu_va_kernel                  NV_ALIGN_BYTES(8);   // OUT: privileged diagnostic only
    NvU64           owner_static_bar1_start        NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_static_bar1_size         NV_ALIGN_BYTES(8);   // OUT
    NvU64           fb_size                        NV_ALIGN_BYTES(8);   // OUT
    NvU64           fb_max_allocatable_address     NV_ALIGN_BYTES(8);   // OUT

    NV_STATUS       acquire_status;                                     // OUT
    NV_STATUS       map_status;                                         // OUT
    NV_STATUS       verify_status;                                      // OUT
    NV_STATUS       release_status;                                     // OUT
    NV_STATUS       rmStatus;                                           // OUT
} UVM_GLM53_RM_CPU_MAP_PROBE_PARAMS;

//
// GLM53 RM CPU mapping physical-address and owner-CE verification probe.
//
#define UVM_GLM53_RM_CPUMAP_PHYS_PROBE                                UVM_IOCTL_BASE(89)
#define UVM_GLM53_RM_CPUMAP_PHYS_PROBE_OP_QUERY                       1U
#define UVM_GLM53_RM_CPUMAP_PHYS_PROBE_OP_ACQUIRE                     2U
#define UVM_GLM53_RM_CPUMAP_PHYS_PROBE_OP_RELEASE                     3U
#define UVM_GLM53_RM_CPUMAP_PHYS_PROBE_OP_VERIFY                      4U

#define UVM_GLM53_RM_CPUMAP_PHYS_VA_CLASS_UNKNOWN                     0U
#define UVM_GLM53_RM_CPUMAP_PHYS_VA_CLASS_VMALLOC                     1U

#define UVM_GLM53_RM_CPUMAP_PHYS_BAR_NONE                             0xffffffffU

typedef struct
{
    NvU32           words_checked;                                      // OUT
    NvU32           mismatches;                                         // OUT
    NvU32           first_mismatch_index;                               // OUT
    NvU32           reserved0;                                          // OUT
    NvU32           first_expected;                                     // OUT
    NvU32           first_observed;                                     // OUT
    NvU32           last_expected;                                      // OUT
    NvU32           last_observed;                                      // OUT
    NV_STATUS       status;                                             // OUT
} UVM_GLM53_RM_CPUMAP_PHYS_COMPARE_RESULT;

typedef struct
{
    NvProcessorUuid gpu_uuid;                                           // IN
    NvU32           op;                                                 // IN
    NvU32           verify_epoch;                                       // IN
    NvU64           length                         NV_ALIGN_BYTES(8);   // IN
    NvU64           gpu_alignment                  NV_ALIGN_BYTES(8);   // IN
    NvU64           seed                           NV_ALIGN_BYTES(8);   // IN
    NvU64           mapping_id                     NV_ALIGN_BYTES(8);   // IN/OUT

    NvU32           active;                                             // OUT
    NvU32           free_completed;                                     // OUT
    NvU32           cpu_va_class;                                       // OUT
    NvU32           cpu_page_count;                                     // OUT
    NvU32           cpu_phys_contiguous;                                // OUT
    NvU32           cpu_phys_segments;                                  // OUT
    NvU32           matched_bar_index;                                  // OUT
    NvU32           matched_bar_full_coverage;                          // OUT

    NvU64           active_length                  NV_ALIGN_BYTES(8);   // OUT
    NvU64           target_gpu_va                  NV_ALIGN_BYTES(8);   // OUT: UVM internal GPU VA
    NvU64           target_cpu_va_kernel           NV_ALIGN_BYTES(8);   // OUT: privileged diagnostic only
    NvU64           verify_gpu_va                  NV_ALIGN_BYTES(8);   // OUT: UVM internal GPU VA
    NvU64           verify_cpu_va_kernel           NV_ALIGN_BYTES(8);   // OUT: privileged diagnostic only
    NvU64           cpu_page_offset                NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_resource_start             NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_resource_end               NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_page0_resource             NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_page1_resource             NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_last_page_resource         NV_ALIGN_BYTES(8);   // OUT
    NvU64           matched_bar_start              NV_ALIGN_BYTES(8);   // OUT
    NvU64           matched_bar_end                NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_offset          NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_start               NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_end                 NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_start               NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_end                 NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_start               NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_end                 NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_bus_start           NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_bus_end             NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_bus_start           NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_bus_end             NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_bus_start           NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_bus_end             NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_bus_start       NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_bus_end         NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_bus_offset      NV_ALIGN_BYTES(8);   // OUT
    NvU32           allocation_bar_bus_valid;                           // OUT
    NvU32           allocation_bar_resource_equals_bus;                 // OUT
    NvU64           owner_static_bar1_start        NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_static_bar1_size         NV_ALIGN_BYTES(8);   // OUT
    NvU64           fb_size                        NV_ALIGN_BYTES(8);   // OUT
    NvU64           fb_max_allocatable_address     NV_ALIGN_BYTES(8);   // OUT

    NV_STATUS       acquire_status;                                     // OUT
    NV_STATUS       target_map_status;                                  // OUT
    NV_STATUS       verify_alloc_status;                                // OUT
    NV_STATUS       verify_map_status;                                  // OUT
    NV_STATUS       phys_query_status;                                  // OUT
    NV_STATUS       ce_forward_status;                                  // OUT
    NV_STATUS       ce_reverse_status;                                  // OUT
    NV_STATUS       release_status;                                     // OUT

    UVM_GLM53_RM_CPUMAP_PHYS_COMPARE_RESULT forward;                   // OUT
    UVM_GLM53_RM_CPUMAP_PHYS_COMPARE_RESULT reverse;                   // OUT

    NV_STATUS       rmStatus;                                           // OUT
} UVM_GLM53_RM_CPUMAP_PHYS_PROBE_PARAMS;

//
// GLM53 serialized BAR1-backed requester read probe.
//
#define UVM_GLM53_SERIAL_BAR1_PEER_READ                               UVM_IOCTL_BASE(90)
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY                       1U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_ACQUIRE_OWNER               2U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_PREPARE_OWNER_EPOCH         3U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_MAP_REQUESTER           4U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_DMA_UNMAP_REQUESTER         5U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_APPLY                 6U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_REMAP_RESTORE               7U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_RELEASE_OWNER               8U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_VERIFY_OWNER_EPOCH          9U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_QUERY_REQUESTER_PTE        10U
#define UVM_GLM53_SERIAL_BAR1_PEER_READ_OP_ACQUIRE_EXTERNAL_OWNER     11U
#define UVM_GLM53_SERIAL_BAR1_DMA_API_DIRECT_DMA_MAP_RESOURCE          1U

typedef struct
{
    NvProcessorUuid requester_uuid;                                     // IN
    NvProcessorUuid owner_uuid;                                         // IN
    NvU32           op;                                                 // IN
    NvU32           epoch;                                              // IN
    NvU64           length                         NV_ALIGN_BYTES(8);   // IN: owner allocation/window length
    NvU64           gpu_alignment                  NV_ALIGN_BYTES(8);   // IN: owner RM allocation alignment
    NvU64           seed                           NV_ALIGN_BYTES(8);   // IN: 0 selects default
    NvU64           mapping_id                     NV_ALIGN_BYTES(8);   // IN/OUT: owner lifetime id
    NvU64           route_id                       NV_ALIGN_BYTES(8);   // IN/OUT: requester DMA route id
    NvU64           test_word_offset               NV_ALIGN_BYTES(8);   // IN: offset inside owner/requester leaf

    NvU32           owner_active;                                       // OUT
    NvU32           dma_active;                                         // OUT
    NvU32           remap_active;                                       // OUT
    NvU32           free_completed;                                     // OUT
    NvU32           cpu_va_class;                                       // OUT
    NvU32           cpu_page_count;                                     // OUT
    NvU32           cpu_phys_contiguous;                                // OUT
    NvU32           cpu_phys_segments;                                  // OUT
    NvU32           matched_bar_index;                                  // OUT
    NvU32           matched_bar_full_coverage;                          // OUT

    NvU64           active_length                  NV_ALIGN_BYTES(8);   // OUT
    NvU64           target_gpu_va                  NV_ALIGN_BYTES(8);   // OUT: owner UVM internal GPU VA
    NvU64           target_cpu_va_kernel           NV_ALIGN_BYTES(8);   // OUT: privileged diagnostic only
    NvU64           verify_gpu_va                  NV_ALIGN_BYTES(8);   // OUT: owner UVM internal GPU VA
    NvU64           verify_cpu_va_kernel           NV_ALIGN_BYTES(8);   // OUT: privileged diagnostic only
    NvU64           cpu_page_offset                NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_resource_start             NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_resource_end               NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_page0_resource             NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_page1_resource             NV_ALIGN_BYTES(8);   // OUT
    NvU64           cpu_last_page_resource         NV_ALIGN_BYTES(8);   // OUT
    NvU64           matched_bar_start              NV_ALIGN_BYTES(8);   // OUT
    NvU64           matched_bar_end                NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_offset          NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_start               NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_end                 NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_start               NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_end                 NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_start               NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_end                 NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_bus_start           NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar0_bus_end             NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_bus_start           NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar1_bus_end             NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_bus_start           NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_bar3_bus_end             NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_bus_start       NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_bus_end         NV_ALIGN_BYTES(8);   // OUT
    NvU64           allocation_bar_bus_offset      NV_ALIGN_BYTES(8);   // OUT
    NvU32           allocation_bar_bus_valid;                           // OUT
    NvU32           allocation_bar_resource_equals_bus;                 // OUT
    NvU64           owner_static_bar1_start        NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_static_bar1_size         NV_ALIGN_BYTES(8);   // OUT
    NvU64           fb_size                        NV_ALIGN_BYTES(8);   // OUT
    NvU64           fb_max_allocatable_address     NV_ALIGN_BYTES(8);   // OUT

    NvU64           requester_dma_mask             NV_ALIGN_BYTES(8);   // OUT
    NvU64           requester_coherent_dma_mask    NV_ALIGN_BYTES(8);   // OUT
    NvU64           requester_dma_addressable_start NV_ALIGN_BYTES(8);  // OUT
    NvU64           requester_dma_addressable_limit NV_ALIGN_BYTES(8);  // OUT
    NvU64           mapped_resource_start          NV_ALIGN_BYTES(8);   // OUT
    NvU64           mapped_resource_end            NV_ALIGN_BYTES(8);   // OUT
    NvU64           requester_dma_address          NV_ALIGN_BYTES(8);   // OUT
    NvU64           requester_dma_length           NV_ALIGN_BYTES(8);   // OUT
    NvU64           destination_gpu_sys_address    NV_ALIGN_BYTES(8);   // OUT
    NvU32           dma_direction;                                      // OUT
    NvU32           dma_mapping_error;                                  // OUT
    NvU32           dma_contiguous_for_length;                          // OUT
    NvU32           dma_aligned_for_128k_leaf;                          // OUT
    NvU32           dma_in_requester_window;                            // OUT
    NvU32           dma_aligned_for_4k_leaf;                             // OUT
    NvU32           dma_api_path;                                       // OUT
    NvU32           linux_dma_map_resource_used;                        // OUT
    NvU32           nv_dma_map_peer_used;                               // OUT
    NvU32           dma_direction_allows_device_read;                   // OUT

    NvU32           root_phys_aperture;                                 // IN
    NvU32           retained_gpu_va_space_state;                        // OUT
    NvU64           va_space_ptr                   NV_ALIGN_BYTES(8);   // IN
    NvU64           gpu_va_space_ptr               NV_ALIGN_BYTES(8);   // IN
    NvU64           root_phys_address              NV_ALIGN_BYTES(8);   // IN
    NvU64           target_va                      NV_ALIGN_BYTES(8);   // IN
    NvU64           page_size                      NV_ALIGN_BYTES(8);   // IN
    NvU64           target_pte_phys                NV_ALIGN_BYTES(8);   // IN
    NvU64           expected_current_pte           NV_ALIGN_BYTES(8);   // IN
    NvU64           requested_temporary_pte        NV_ALIGN_BYTES(8);   // IN: exact diagnostic PTE, 0 uses DMA route
    NvU64           requested_destination_gpu_sys_address NV_ALIGN_BYTES(8); // IN: expected decoded sys address for custom PTE
    NvU32           requested_allow_writable;                           // IN: only for local host-mapped control
    NvU32           reserved0;                                          // IN/OUT
    NvU64           pte_before                     NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_written                    NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_after                      NV_ALIGN_BYTES(8);   // OUT
    NvU64           saved_original_pte             NV_ALIGN_BYTES(8);   // OUT
    NvU64           temporary_pte                  NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_pte_phys         NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_va               NV_ALIGN_BYTES(8);   // OUT
    NvU64           decoded_sys_address            NV_ALIGN_BYTES(8);   // OUT
    NvU32           decoded_aperture_bits;                              // OUT
    NvU32           decoded_kind;                                       // OUT
    NvU32           decoded_read_only;                                  // OUT
    NvU32           decoded_read_disable;                               // OUT
    NvU32           decoded_write_disable;                              // OUT
    NvU32           decoded_volatile;                                   // OUT

    NvU32           expected_word;                                      // OUT
    NvU32           observed_word_after_owner_ce;                       // OUT
    NvU32           owner_cpu_readb_expected;                           // OUT
    NvU32           owner_cpu_readb_observed;                           // OUT
    NvU32           owner_cpu_readb_match;                              // OUT
    NvU32           reserved1;                                          // OUT
    NvU64           owner_cpu_readb_offset         NV_ALIGN_BYTES(8);   // OUT
    NvU64           owner_cpu_readb_resource       NV_ALIGN_BYTES(8);   // OUT

    NV_STATUS       acquire_status;                                     // OUT
    NV_STATUS       target_map_status;                                  // OUT
    NV_STATUS       verify_alloc_status;                                // OUT
    NV_STATUS       verify_map_status;                                  // OUT
    NV_STATUS       phys_query_status;                                  // OUT
    NV_STATUS       ce_forward_status;                                  // OUT
    NV_STATUS       dma_map_status;                                     // OUT
    NV_STATUS       dma_unmap_status;                                   // OUT
    NV_STATUS       owner_cpu_readb_status;                             // OUT
    NV_STATUS       requester_l2_invalidate_status;                     // OUT
    NV_STATUS       requester_phys_tlb_invalidate_status;               // OUT
    NV_STATUS       pte_update_status;                                  // OUT
    NV_STATUS       release_status;                                     // OUT

    UVM_GLM53_RM_CPUMAP_PHYS_COMPARE_RESULT owner_forward;             // OUT

    NV_STATUS       rmStatus;                                           // OUT

    // Inputs/outputs for a caller-retained RM-reflected BAR1 mapping. The
    // userspace owner keeps the CUDA/RM allocation and reflected mapping
    // alive; UVM validates this range against the owner's BAR1 resource and
    // retains the owner GPU reference while requester DMA mapping is active.
    NvU64           requested_cpu_resource_start   NV_ALIGN_BYTES(8);   // IN
    NvU64           requested_owner_bar1_bus_start NV_ALIGN_BYTES(8);   // IN
    NvU64           requested_resource_length      NV_ALIGN_BYTES(8);   // IN
    NvU32           owner_external;                                     // OUT
} UVM_GLM53_SERIAL_BAR1_PEER_READ_PARAMS;

//
// GLM53 bounded multi-PTE remap transaction for the M10 multicast/reduce harness.
// This is intentionally narrower than a physical-memory writer: each apply is
// bound to a live GPU VA space/root, an exact leaf address, an expected current
// PTE and a predecoded SYS destination address. Restore uses the saved original
// PTE retained in kernel state.
//
#define UVM_GLM53_MULTI_PTE_REMAP                                      UVM_IOCTL_BASE(91)
#define UVM_GLM53_MULTI_PTE_REMAP_OP_QUERY                             1U
#define UVM_GLM53_MULTI_PTE_REMAP_OP_APPLY                             2U
#define UVM_GLM53_MULTI_PTE_REMAP_OP_RESTORE                           3U
#define UVM_GLM53_MULTI_PTE_REMAP_OP_REAP_DEAD                         4U
#define UVM_GLM53_MULTI_PTE_REMAP_MAX_ACTIVE                           512U

typedef struct
{
    NvProcessorUuid gpu_uuid;                                           // IN
    NvU32           op;                                                 // IN
    NvU32           slot;                                               // OUT
    NvU64           mapping_id                     NV_ALIGN_BYTES(8);   // IN/OUT

    NvU32           active_count;                                       // OUT
    NvU32           root_phys_aperture;                                 // IN
    NvU32           retained_gpu_va_space_state;                        // OUT
    NvU32           reserved0;                                          // IN/OUT

    NvU64           va_space_ptr                   NV_ALIGN_BYTES(8);   // IN
    NvU64           gpu_va_space_ptr               NV_ALIGN_BYTES(8);   // IN
    NvU64           root_phys_address              NV_ALIGN_BYTES(8);   // IN
    NvU64           target_va                      NV_ALIGN_BYTES(8);   // IN
    NvU64           page_size                      NV_ALIGN_BYTES(8);   // IN
    NvU64           target_pte_phys                NV_ALIGN_BYTES(8);   // IN
    NvU64           expected_current_pte           NV_ALIGN_BYTES(8);   // IN
    NvU64           requested_temporary_pte        NV_ALIGN_BYTES(8);   // IN
    NvU64           requested_destination_gpu_sys_address NV_ALIGN_BYTES(8); // IN
    NvU32           requested_allow_writable;                           // IN
    NvU32           reserved1;                                          // IN/OUT

    NvU64           pte_before                     NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_written                    NV_ALIGN_BYTES(8);   // OUT
    NvU64           pte_after                      NV_ALIGN_BYTES(8);   // OUT
    NvU64           saved_original_pte             NV_ALIGN_BYTES(8);   // OUT
    NvU64           temporary_pte                  NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_pte_phys         NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_target_va               NV_ALIGN_BYTES(8);   // OUT
    NvU64           active_destination_gpu_sys_address NV_ALIGN_BYTES(8);// OUT
    NvU32           decoded_aperture_bits;                              // OUT
    NvU32           decoded_kind;                                       // OUT
    NvU32           decoded_read_only;                                  // OUT
    NvU32           decoded_read_disable;                               // OUT
    NvU32           decoded_write_disable;                              // OUT
    NvU32           decoded_volatile;                                   // OUT
    NvU32           duplicate_or_stale;                                 // OUT
    NvU32           pte_update_attempted;                               // OUT
    NvU32           pte_restored;                                       // OUT
    NvU32           reserved2;                                          // OUT

    NV_STATUS       lookup_status;                                      // OUT
    NV_STATUS       read_status;                                        // OUT
    NV_STATUS       requester_l2_invalidate_status;                     // OUT
    NV_STATUS       requester_phys_tlb_invalidate_status;               // OUT
    NV_STATUS       pte_validation_status;                              // OUT
    NV_STATUS       pte_update_status;                                  // OUT
    NV_STATUS       rmStatus;                                           // OUT
} UVM_GLM53_MULTI_PTE_REMAP_PARAMS;

//
// GLM53 bounded cache synchronization diagnostic. This does not alter PTEs or
// mappings. It exposes the UVM L2 invalidate helper for one retained GPU so a
// resident-kernel visibility test can distinguish payload-cache staleness from
// command-publication bugs.
//
#define UVM_GLM53_LOCAL_CACHE_SYNC                                     UVM_IOCTL_BASE(92)
#define UVM_GLM53_LOCAL_CACHE_SYNC_OP_SYNC                             1U
#define UVM_GLM53_LOCAL_CACHE_SYNC_APERTURE_VID                        1U
#define UVM_GLM53_LOCAL_CACHE_SYNC_APERTURE_SYS                        2U
#define UVM_GLM53_LOCAL_CACHE_SYNC_APERTURE_SYS_NON_COHERENT           3U

typedef struct
{
    NvProcessorUuid gpu_uuid;                                           // IN
    NvU32           op;                                                 // IN
    NvU32           aperture;                                           // IN
    NvU32           do_phys_tlb_invalidate;                             // IN

    NvU32           retained_gpu;                                       // OUT
    NvU32           reserved0;                                          // OUT

    NV_STATUS       l2_invalidate_status;                               // OUT
    NV_STATUS       phys_tlb_invalidate_status;                         // OUT
    NV_STATUS       rmStatus;                                           // OUT
} UVM_GLM53_LOCAL_CACHE_SYNC_PARAMS;

//
// Temporary ioctls which should be removed before UVM 8 release
// Number backwards from 2047 - highest custom ioctl function number
// windows can handle.
//

//
// UvmIs8Supported
//
#define UVM_IS_8_SUPPORTED                                            UVM_IOCTL_BASE(2047)

typedef struct
{
    NvU32     is8Supported; // OUT
    NV_STATUS rmStatus;     // OUT
} UVM_IS_8_SUPPORTED_PARAMS;

#ifdef __cplusplus
}
#endif

#endif // _UVM_IOCTL_H
