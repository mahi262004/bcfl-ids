# IoT Device Layer -- Phase 2

Confirmed per synopsis: physical hardware testbed using a
Raspberry Pi 4/5 as edge gateway, a smart IP security camera, and a
Wi-Fi router.

This folder is for Phase 2: once the pipeline is proven on the real
N-BaIoT dataset (Phase 1 -- fl/, blockchain/, xai/), extend it to
your own physical device traffic.

Note: N-BaIoT itself already includes real Provision/SimpleHome
security camera devices among its 9 devices, so your physical
testbed's camera traffic is a natural, literature-consistent
extension of the same device category -- not a different domain.

Planned contents once Phase 2 starts:
- Traffic capture scripts (e.g. via Wireshark/tshark) from the
  physical camera + Pi setup
- Feature extraction matching N-BaIoT's statistical feature style,
  so the same fl/model.py and fl/data_loader.py conventions apply
  to real captured traffic
- Notes on how this data plugs into fl/client.py as an additional,
  real (10th) client
