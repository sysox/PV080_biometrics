#!/bin/sh
LIST_OF_APPS="cups-pdf python3-pip default-jre libreoffice-java-common"

apt update

apt install -y $LIST_OF_APPS

pip3 install python-docx
