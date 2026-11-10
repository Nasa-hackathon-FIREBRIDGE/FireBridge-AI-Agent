#!/usr/bin/env bash
set -euo pipefail
umask 077
revision=${RELEASE_REVISION:?}
source_image=${AGENT_SOURCE:?Verified GHCR digest required}
[[ "$revision" =~ ^[a-f0-9]{40}$ ]] || exit 1
[[ "$source_image" =~ ^ghcr.io/nasa-hackathon-firebridge/firebridge-ai-agent@sha256:[a-f0-9]{64}$ ]] || exit 1
[[ "${DEPLOY_HOST:?}" =~ ^[a-zA-Z0-9.-]+$ ]] || exit 1
temporary=$(mktemp -d)
trap 'rm -rf "$temporary"' EXIT
printf '%s\n' "${DEPLOY_SSH_KEY:?}" > "$temporary/key"
printf '%s\n' "${DEPLOY_KNOWN_HOSTS:?}" > "$temporary/known_hosts"
chmod 600 "$temporary/key" "$temporary/known_hosts"
docker pull "$source_image"
docker tag "$source_image" "firebridge-ai-agent:$revision"
docker save "firebridge-ai-agent:$revision" | gzip -1 > "$temporary/image.tar.gz"
checksum=$(sha256sum "$temporary/image.tar.gz" | cut -d ' ' -f 1)
options=(-F /dev/null -i "$temporary/key" -o IdentitiesOnly=yes -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$temporary/known_hosts" -o ConnectTimeout=15 -o ServerAliveInterval=20 -o ServerAliveCountMax=15)
target="firebridge-agent-deploy@$DEPLOY_HOST"
ssh "${options[@]}" "$target" "image $revision $checksum" < "$temporary/image.tar.gz"
ssh "${options[@]}" "$target" "activate $revision"
ssh "${options[@]}" "$target" status
