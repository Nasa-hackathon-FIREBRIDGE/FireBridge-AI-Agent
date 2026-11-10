# FireBridge AI Agent production

The private repository deploys `main` automatically to the existing Contabo host.
The API base is `https://astramind.solutions/agent/v1`. Clients send `X-API-Key`.
Liveness and readiness are `/agent/health` and `/agent/ready`. The application
refuses to start in production without both service authentication and a Groq key.
Readiness checks SQLite, a configured Groq key and model availability in the local
router; it does not make a billable provider call or validate the provider key.

## Pipeline

Every pull request runs locked Python 3.12 dependency installation, Ruff, pytest,
deployment shell checks, source credential scanning, a production image build,
HIGH/CRITICAL vulnerability scanning (including unfixed findings), and container
checks for readiness, authentication and persistence across restart.

Successful `main` builds publish the exact tested image to GHCR with a commit tag.
The deploy job selects its immutable digest, transfers a checksum-verified archive
over pinned-host SSH, and activates it through a forced-command account. Pull
requests cannot use production credentials or deploy. Queued older revisions are
skipped if `main` has moved forward. Actions dependencies are pinned by commit;
Dependabot proposes weekly Actions, Docker and Python dependency updates.

GitHub environment `agent-production` contains `DEPLOY_SSH_KEY` and
`DEPLOY_KNOWN_HOSTS`, plus variable `DEPLOY_HOST`. Its deployment policy permits
only `main`; no manual reviewer is configured, so deployment is automatic.
Repository variable `AGENT_DEPLOY_ENABLED=true` enables the production job.

## Server layout and recovery

The independent Compose project lives at `/opt/firebridge/agent`:

- `runtime.env`: root-only provider configuration and service API key.
- `data/agent.db`: SQLite analyses, conversations and memories, owned by UID 10001.
- `.env`: selected image reference; contains no application credentials.
- `deployment.json`: current revision, previous revision and backup receipt.
- `backups/`: SQLite online backups including committed WAL data; latest 14 retained.
- `operations.log`: private deployment failures.

Only `127.0.0.1:18000` is published. The existing HTTPS proxy reaches the agent via
`firebridge-webapp_web` and the `firebridge-ai-agent` network alias. Its `/agent/`
route uses Docker DNS at request time, allowing container replacement without
stale addresses. `/agent/v1/*` remains key protected. A service key grants access
to the agent's shared sessions and memories; keep it in trusted server clients,
not frontend JavaScript. End-user identity and per-user authorization require a
separate application authentication design.

Activation backs up the database, checks image source/revision/non-root identity,
waits for container readiness, then verifies the running revision and both
authenticated and unauthenticated API behavior. On activation failure, the
previous image is restored and checked. The first failed activation stops the
candidate and preserves its data directory. Existing images are retained for
rollback; neither shared Docker images nor volumes are automatically pruned.

Deployments can briefly interrupt requests while the single process restarts.
Image rollback preserves current data; incompatible schema changes require an
administrator to restore a backup. Backups are local to this server, so off-site
backup remains a separate operational concern. Failures detected only by the
public HTTPS check fail the workflow; the server receipt records the successful
internal activation and lets an administrator select the previous revision.

```bash
# On the VPS as an administrator:
cd /opt/firebridge/agent
docker compose -f compose.yaml ps
docker compose -f compose.yaml logs --tail=100 agent
cat deployment.json
curl --fail http://127.0.0.1:18000/ready

# Roll back an already loaded, verified revision:
SSH_ORIGINAL_COMMAND='rollback <40-character-commit>' /usr/local/sbin/firebridge-agent-gateway

# Read the service API key privately on the server; never paste it into Git or chat:
sudo python3 -c 'from pathlib import Path; import json; e=dict(x.split("=",1) for x in Path("/opt/firebridge/agent/runtime.env").read_text().splitlines() if "=" in x); print(json.loads(e["FBA_SERVICE_API_KEYS"])[0])'
```

## First installation on the existing FireBridge server

1. Preserve the running webapp, TLS certificates and private database. Verify
   Docker Compose supports raw env files and `up --wait`, and the external
   `firebridge-webapp_web` network exists.
2. Generate a dedicated Ed25519 CI key outside the repository. Prepare a root-only
   runtime env file with `FBA_GROQ_API_KEY`, JSON `FBA_SERVICE_API_KEYS`,
   `FBA_MODELS`, and `FBA_FIREBRIDGE_API_URL` as needed. Never upload runtime
   credentials to build jobs or include `.env` in an image.
3. Copy this `deploy/` directory to the host. Run as root:
   `python3 deploy/provision_server.py --public-key /private/ci.pub --runtime-env /private/runtime.env`.
   Re-running preserves existing runtime credentials. Changes to infrastructure
   scripts require an administrator install; routine deployments update images.
4. Add the environment, its secrets and host variable to GitHub; permit `main`
   only. Upload verified SSH host keys rather than accepting a fresh key in CI.
5. Run `python3 deploy/install_ingress.py` on the server and preserve the same
   route in the webapp's `deploy/nginx.proxy.conf` source. The installer validates
   Nginx before reload and restores the prior config if validation fails.
6. Enable `AGENT_DEPLOY_ENABLED`, push `main`, and verify all three jobs and the
   public readiness revision. Groq credentials must also be verified separately.

The SSH account `firebridge-agent-deploy` has no unrestricted sudo, interactive
shell, forwarding, Docker group membership, database access, or data-stack control.
It accepts only `status`, `image <sha> <checksum>`, `activate <sha>`, and
`rollback <sha>`. The checked-in gateway and Compose configuration are root owned.
