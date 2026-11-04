# Late-power boot

Brings up the five M10 cards on the second PSU after the BIOS has booted with only one card powered. Why this is necessary: [docs/HARDWARE.md](../docs/HARDWARE.md#why-the-cards-are-powered-late).

```bash
bash boot/build-sanity.sh                        # builds the CUDA check program (once)
sudo boot/prepare-and-suspend.sh --check-only    # dry run of the checks
sudo boot/prepare-and-suspend.sh                 # PSU2 must be OFF; enters deep S3
# while asleep: PSU2 on, wait ~5 s, press the power button once
sudo boot/resume-and-enumerate.sh
```

`common.sh` holds the shared logic. The late ports are discovered from sysfs (PEX8748 downstream ports in slots 0x09–0x13, the slot rule of the kernel quirk), or taken from `LATE_PORTS_OVERRIDE`. Results go to `boot/results/<timestamp>/`. Close `nvtop` and every CUDA client before the prepare step. Never switch PSU2 off while late devices are registered.
