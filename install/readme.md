# Futronic Fingerprint Scanner + Jupyter Setup (PV080_biometrics)

## 1. Install system dependencies
```bash
sudo apt update
sudo dpkg --add-architecture i386
sudo apt update
sudo apt install libusb-0.1-4 libusb-dev \
                 libc6:i386 libstdc++6:i386 \
                 libusb-0.1-4:i386 libgtk2.0-0:i386
```

## 2. Enable USB access (FIX sudo-only issue)
If scanner works with `sudo` but not without it → **udev permissions problem**.

```bash
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="1491", ATTR{idProduct}=="0020", MODE="0666"' | sudo tee /etc/udev/rules.d/99-futronic.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

👉 Unplug + replug scanner

Verify device:
```bash
lsusb
```

⚠️ If still not working:
- Check correct vendor ID (e.g. `0fca` vs `0f1a`)
- Update rule accordingly

---

## 3. Prepare runtime
```bash
cd ~/PycharmProjects/PV080_biometrics/install
rm -f libusb.so
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH
```

---

## 4. Test scanner
```bash
chmod +x ftrScanAPI_Ex gtk_ex
./ftrScanAPI_Ex
```

Expected output:
```
Image size is 153600
Please put your finger on the scanner:
Capturing fingerprint ......
Done!
Writing to file......
Fingerprint image is written to file: frame_Ex.bmp.
```

⚠️ If only this works:
```bash
sudo ./ftrScanAPI_Ex
```
➡ Then udev rules are still incorrect.

---

## 5. Optional GUI demo
```bash
./gtk_ex
```

---

## 6. Install Conda + Jupyter
```bash
cd ..
# Create environment
conda create -n biometrics python=3.11 -y
conda activate biometrics

# Upgrade pip
python -m pip install --upgrade pip

# Install from requirements file
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user \
    --name biometrics \
    --display-name "Python (biometrics)"

# Launch Jupyter
jupyter notebook

---

## ✅ DONE when
- Scanner works **without sudo**
- Fingerprint image is captured (`frame_Ex.bmp`)
- Jupyter successfully loads `libScanAPI.so`
