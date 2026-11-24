# KB-002: Password reset and multi-factor authentication

- Owner: Identity and Access Management
- Last reviewed: 2026-08-20
- Status: Active
- Applies to: Company-managed identities

Send the user to the self-service password reset portal. Neither service desk
staff nor automated agents may ever request or keep a password. When the
self-service reset does not work, open an `Identity` ticket that records the
error and the affected username, and leave out any authentication secrets.

Users register a company-approved authenticator in the identity portal.
An MFA code must never be requested, repeated, logged, or validated, whether
in chat or in a ticket. Before MFA methods are reset for a lost or replaced
device, a human service desk agent must verify the user's identity.

Treat repeated MFA prompts the user did not expect, or a reset request the user
did not start, as a potential security incident and route it to Security.
