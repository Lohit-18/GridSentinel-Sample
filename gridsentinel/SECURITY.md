# Security Policy

## Scope

GridSentinel is a synthetic, offline research project. It does not connect to real
networks or devices, so its attack surface is limited to the Python package
itself (e.g. unsafe handling of a crafted input file).

## Reporting a vulnerability

If you find a security issue in the **code** (for example, a parsing bug in the
GOOSE decoder that could crash or execute on a malicious input):

1. **Do not** open a public issue for anything exploitable.
2. Email the maintainer or use GitHub's **private vulnerability reporting**
   ("Security" tab → "Report a vulnerability").
3. Include steps to reproduce and the affected version.

You can expect an acknowledgement within a few days.

## Out of scope

- Reports about attacking real power grids or other infrastructure. This project
  does not provide that capability and will not add it. See `ETHICS.md`.
- The detector's statistical performance on your own data — open a normal issue
  for that.

## Supported versions

The latest `main` branch and the most recent tagged release receive fixes.
