# Ethics & Responsible Use

GridSentinel is a **defensive** research and education project. Its purpose is to
*detect* attacks on power-grid monitoring, so that operators and researchers can
study and improve grid resilience.

## What this project is

- A **fully synthetic simulation**. Every grid, every measurement and every
  attack is generated in software from standard IEEE test systems. Running it
  touches no real substation, no real device, and no real network.
- The attack generators (`gridsentinel.attacks`) exist for **one reason only**:
  to produce labelled examples so the detector can learn to recognise them. They
  operate on the in-memory simulation and have no capability to reach, probe,
  scan, or transmit to any external system.
- The GOOSE module models the IEC 61850 message *format* for inspection and
  feature extraction. It does not publish, sniff, or inject traffic on any live
  network.

## What this project is not

- It is **not** a tool for attacking power grids or any other infrastructure, and
  it must not be adapted for that purpose. Interfering with electricity
  infrastructure endangers lives and is a serious crime in virtually every
  jurisdiction.
- It is **not** a certified intrusion-detection system or protection relay. Do
  not deploy it to protect real infrastructure without independent validation,
  domain review, and appropriate engineering safeguards.

## If you build on it

- Keep the defensive framing. Contributions that add offensive capability against
  real systems (live traffic injection, device exploitation, evasion tooling
  aimed at real deployments) are out of scope and will not be accepted.
- Only ever test against systems you own or are explicitly authorised to assess.
- Follow responsible-disclosure practices for any real-world vulnerability you
  discover elsewhere.

By using this project you agree to use it lawfully and ethically.
