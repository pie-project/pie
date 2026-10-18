#!/bin/bash
# $1 = cluster policy
R=/root/Workspace/pie/target/release; cd /root/boise
for n in pie-worker pie-controller pie-gateway; do pgrep -x $n | xargs -r kill; done
timeout 120 bash -c 'while pgrep -x pie-worker >/dev/null || pgrep -x pie-gateway >/dev/null || pgrep -x pie-controller >/dev/null; do sleep 2; done'
sed "s/policy = \"pin\"/policy = \"$1\"/" gw.toml > gwr.toml
nohup $R/pie-controller --listen 127.0.0.1:7000 --metrics-addr 127.0.0.1:9300 > ctl.log 2>&1 &
sleep 1; nohup $R/pie-gateway -c gwr.toml --metrics-addr 127.0.0.1:9301 --log-level "info,gateway=debug" > gw.log 2>&1 &
sleep 2
nohup $R/pie-worker -c wa4.toml --metrics-addr 127.0.0.1:9310 > wa.log 2>&1 &
nohup $R/pie-worker -c wb4.toml --metrics-addr 127.0.0.1:9311 > wb.log 2>&1 &
timeout 400 bash -c 'until sed "s/\x1b\[[0-9;]*m//g" gw.log | grep -aq "routing table updated.*workers=2"; do sleep 3; done'
sleep 5
sed 's/\x1b\[[0-9;]*m//g' wa.log | grep -a -o 'worker_id=worker#[0-9]*' | head -1 | cut -d'#' -f2
