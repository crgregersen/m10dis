# Arctic Fan Controller DKMS driver backported for Ubuntu 24.04

## Purpose

This repository backports the Linux kernel HWMON driver
`drivers/hwmon/arctic_fan_controller.c` from upstream Linux into an Ubuntu
system as an out-of-tree DKMS module. It also integrates patches from LKML.

The architectural separation of concerns is:

- **Git** tracks the pristine upstream source and project configuration.
- **Quilt** tracks the downstream patch series (compatibility fixes, etc.).
- The Quilt-patched source tree (`build/`) is generated and disposable — it is
  **never** committed to Git.
- **DKMS** builds the patched source for each installed kernel.
- LKML patches are retrieved **manually** by the user with `b4`; the project
  never downloads patches automatically.

The key invariant: **never edit or commit the generated patched source as the
authoritative source.** The authoritative state is always:

```
upstream source + patches/series + patch files + DKMS/build scripts
```

A fresh clone of this repository is sufficient to recreate the patched source
tree deterministically.

```
                   ┌──────────────────────────┐
                   │   Upstream Linux source  │
                   │ arctic_fan_controller.c  │
                   └────────────┬─────────────┘
                                │ pristine copy
                                v
                   ┌──────────────────────────┐
                   │         build/           │
                   │  disposable source tree  │
                   └────────────┬─────────────┘
                                │ quilt push -a
                                v
                   ┌──────────────────────────┐
                   │  patched source tree     │
                   └────────────┬─────────────┘
                                │
                                v
                   ┌──────────────────────────┐
                   │          DKMS            │
                   │ build against each       │
                   │ installed kernel         │
                   └────────────┬─────────────┘
                                │
                                v
                   arctic_fan_controller.ko
```

And independently:

```
LKML / lore.kernel.org
          │ manual retrieval (b4)
          v
     patch file
          │
          v
      patches/
          │
          v
    Quilt patch stack
```

## Upstream source

```
Upstream source: Linux
Path: drivers/hwmon/arctic_fan_controller.c
Commit: e28d0c73d4d7adc9cd3747d81fdc7338217f9a0c
Subject: hwmon: add driver for ARCTIC Fan Controller
Date: 2026-05-08
Link: https://lore.kernel.org/r/20260508064405.38676-1-aureo.serrano@arctic.de
```

The file `upstream/arctic_fan_controller.c` is the pristine upstream baseline.
Do not make compatibility modifications directly to this file — represent them
as Quilt patches instead.

## Repository structure

```
arctic-fan-controller/
├── upstream/          pristine upstream driver source
├── patches/           downstream Quilt patch queue (series is authoritative)
├── build/             generated patched source (disposable, never committed)
├── dkms/              DKMS configuration and generated source snapshot
├── scripts/           prepare.sh, build.sh, clean.sh, dkms-install.sh, ...
├── Makefile           out-of-tree kernel module Makefile
├── .gitignore
└── README.md
```

## Prerequisites

```
sudo apt install build-essential linux-headers-$(uname -r) quilt dkms
```

`b4` is only needed for manually obtaining LKML patches (see below) and is
**not** a build or runtime dependency.

## Getting the LKML patches

Patches from the Linux kernel mailing list are retrieved **manually**. No LKML
subscription is necessary — everything is available on
[lore.kernel.org](https://lore.kernel.org).

Install `b4`:

```
sudo apt install b4
```

General workflow:

1. Find the patch or series on Lore (e.g. via
   <https://lore.kernel.org/all/>).
2. Retrieve it with `b4`.
3. Save the resulting patch/mailbox.
4. Import the relevant patch into the Quilt stack.
5. Preserve the original patch headers and commit message.
6. Add the patch filename to `patches/series`.
7. Apply the stack with `./scripts/prepare.sh`.
8. Resolve compatibility problems in `build/`.
9. Use `quilt refresh` to update the downstream patch.

For a **single patch**, the raw message can be fetched directly from Lore,
e.g.:

```
curl -o patches/0001-some-fix.patch \
    https://lore.kernel.org/all/<message-id>/raw
```

For a **patch series**, prefer `b4` because it understands mailing-list
threading and patch revisions:

```
b4 am <message-id> -o /tmp/patches
```

Note that `b4 am` retrieves and prepares a patch series as an mbox but does
**not** modify the current Git branch. This is exactly what we want: it
obtains the original mailing-list patches without turning Git into the
project's patch-stack manager.

Import a patch from the mbox into the Quilt stack, preserving its original
headers (author, commit message, `Signed-off-by`, `Reviewed-by`, `Acked-by`,
`Link`, `Message-ID`, version info such as `[PATCH v2]`):

```
cd build
quilt import /tmp/patches/0001-some-fix.patch
quilt push
```

Do not rewrite those provenance headers unnecessarily — they document where
the patch came from and who reviewed it.

## Quilt patch workflow

The authoritative patch ordering is `patches/series`. Each patch is a normal
patch file. Quilt metadata lives inside `build/` (`.pc/`), so it never
pollutes Git.

Apply all patches:

```
./scripts/prepare.sh
```

Inspect the stack:

```
cd build
quilt applied
quilt series
```

Remove the top patch:

```
quilt pop
```

Apply the next patch:

```
quilt push
```

Modify the currently patched source and update the patch:

```
# edit files in build/ ...
quilt refresh
```

`quilt refresh` writes the change back to the corresponding file under
`patches/` — not as a modified source file. After changing a patch, only the
patch file (and possibly `patches/series`) should normally be committed to
Git.

## Building manually

Happy path:

```
./scripts/prepare.sh
make
```

`make` builds against `/lib/modules/$(uname -r)/build`. To target a different
kernel:

```
make KDIR=/lib/modules/<version>/build
```

Clean up:

```
make clean        # or: ./scripts/clean.sh
```

## Installing with DKMS

DKMS builds the source **after** the Quilt patch stack has been applied. The
DKMS source snapshot is generated from `upstream/` + `patches/` at install
time, so a DKMS rebuild (e.g. after a kernel update) never depends on a
developer's temporary `build/` directory, Git, `b4`, or network access.

Install:

```
sudo ./scripts/dkms-install.sh
```

This:

1. Generates a self-contained patched source snapshot in `dkms/src/`
   (git-ignored).
2. Registers it with DKMS as `arctic-fan-controller/0.1.0`.
3. Builds and installs `arctic_fan_controller.ko` for the running kernel.
4. Configures the module to load automatically at boot via
   `/etc/modules-load.d/arctic-fan-controller.conf`.

Because `AUTOINSTALL="yes"` is set in `dkms/dkms.conf`, DKMS automatically
rebuilds the module for newly installed kernels.

Load the module right away (subsequent boots load it automatically):

```
sudo modprobe arctic_fan_controller
```

Verify:

```
lsmod | grep arctic
dmesg | tail
```

Uninstall cleanly:

```
sudo ./scripts/dkms-remove.sh
```

This removes the DKMS module and the boot-time autoload entry.

## Updating the upstream driver

1. Fetch the current version of
   [`drivers/hwmon/arctic_fan_controller.c`](https://github.com/torvalds/linux/blob/master/drivers/hwmon/arctic_fan_controller.c)
   from the upstream Linux repository.
2. Replace `upstream/arctic_fan_controller.c`.
3. Update the commit SHA in the "Upstream source" section above (determine it
   from the upstream repository history — do not invent it).
4. Regenerate the build tree: `./scripts/prepare.sh`.
5. Verify that the remaining downstream patches still apply.
6. Commit the upstream source update (and any patch adjustments).

## Updating/replacing LKML patches

When a newer revision of a patch appears on LKML (e.g. `[PATCH v3]` replacing
`0002-some-fix.patch`):

1. Retrieve the newer revision with `b4`.
2. Replace the old patch file (e.g. `quilt import -P 0002-some-fix.patch -f
   <new-file>` or simply overwrite the file).
3. Keep the desired filename/order in `patches/series`, unless there is a
   reason to rename it.
4. Regenerate the build tree: `./scripts/prepare.sh`.
5. Resolve any conflicts.
6. Commit only the new/revised patch and any related metadata.

Do not automatically merge v1/v2/v3 patches together. The patch currently
selected by `patches/series` is the source of truth.

## Dropping patches merged upstream

When a downstream patch is eventually merged upstream:

1. Update `upstream/arctic_fan_controller.c` to a newer upstream Linux version
   containing the change.
2. Remove the now-upstream patch from `patches/series`.
3. Delete the obsolete patch file.
4. Regenerate `build/` (`./scripts/prepare.sh`).
5. Verify that the remaining downstream patches still apply.
6. Commit the upstream source update and patch removal.

Do not leave duplicate patches in the downstream stack if the corresponding
change is already present in the new upstream source.

## Troubleshooting

- **A patch fails to apply during `prepare.sh`**: the script fails loudly on
  purpose. Inspect the rejects in `build/`, fix the patch (e.g. `quilt push`,
  edit, `quilt refresh`), and re-run.
- **`make` fails with missing headers**: install
  `linux-headers-$(uname -r)`.
- **DKMS build fails after a kernel update**: check `dkms status` and
  `/var/lib/dkms/arctic-fan-controller/0.1.0/build/make.log`. The DKMS source
  snapshot may be stale — re-run `sudo ./scripts/dkms-install.sh` to
  regenerate it from the current `upstream/` + `patches/`.
- **Module does not load**: check `dmesg` for USB device enumeration
  (VID `0x3904`, PID `0xF001`).
- **Git status dirty after building**: `build/` and `dkms/src/` are
  git-ignored; if you see them, verify `.gitignore` is intact.

## License

The driver source is licensed `GPL-2.0-or-later OR BSD-2-Clause`. The packaging scripts in this repository
follow the same license.
