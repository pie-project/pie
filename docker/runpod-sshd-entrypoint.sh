#!/bin/sh
# Copied from test-time-bench's backend/docker/runpod-sshd-entrypoint.sh.
#
# RunPod injects the pod's public key as PUBLIC_KEY at container start, so the
# image never bakes an authorized_keys. Refusing to start without it is the point:
# an sshd with no authorized key is a pod that bills and cannot be reached, and
# the failure should name the missing registration rather than look like a
# networking problem.
set -eu

authorized_keys=/root/.ssh/authorized_keys

mkdir -p /run/sshd /root/.ssh
chmod 700 /root/.ssh

if [ -z "${PUBLIC_KEY:-}" ]; then
  echo "ERROR: RunPod PUBLIC_KEY is empty; refusing to start sshd without /root/.ssh/authorized_keys." >&2
  echo "Register the matching public key in the RunPod account before creating the pod." >&2
  exit 1
fi

printf '%s\n' "$PUBLIC_KEY" > "$authorized_keys"
chmod 600 "$authorized_keys"

# The same care the image takes with authorized_keys applies to the host key, which
# is the half that was originally missed. The image ships no /etc/ssh/ssh_host_*,
# because it is public and anonymously pullable: a baked host key would publish the
# private identity every pod presents, and impersonating a pod would raise no
# host-key warning precisely because clients already trust that key. So each
# container mints its own here, before sshd can start. ssh-keygen -A creates only
# what is missing, so a container that restarts keeps the identity it generated.
ssh-keygen -A

if [ "$#" -eq 0 ]; then
  set -- -D -e
fi

exec /usr/sbin/sshd "$@"
