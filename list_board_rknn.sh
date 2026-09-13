#!/usr/bin/env bash
sshpass -p 'yujiarong520' ssh root@192.168.2.27 'find /root/audio8-asr -maxdepth 3 -name "*.rknn" -printf "%p\n"'
