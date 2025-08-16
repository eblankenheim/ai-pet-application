# AI Pet Application: Setup & Deployment Notes

## 1. Python Virtual Environment Setup

### Windows

```cmd
py -m venv venv
venv\Scripts\activate
```

### Linux

```bash
python3 -m venv venv
source venv/bin/activate  # For venv named 'venv'
source myenv313/bin/activate  # For venv named 'myenv313'
```

### Exit Virtual Environment

```bash
deactivate
```

---

## 2. Install Application Requirements

```bash
pip install -r requirements.txt
# Or, for a specific Python executable:
/path/to/python -m pip install -r requirements.txt
```

---

## 3. Running the Flask App

```bash
python app.py
```

---

## 4. Testing the /predict Endpoint

```bash
curl -X POST -F "file=@test.jpg" http://localhost:5000/predict
```

---

## 5. Chalice (Legacy/Alternative)

```bash
cd aipet_chalice
chalice local
```

---

## 6. EC2 Setup & SSH Access

### Install pip on Amazon Linux

```bash
sudo yum install python3-pip -y
```

### Install dependencies on EC2

```bash
pip3 install torch torchvision pillow flask
```

### SSH into EC2

```bash
ssh -i /path/to/ai-pets_key.pem ec2-user@<EC2_PUBLIC_IP>
# Or (if using a shortcut)
ssh ai-pet-application
```

### Exit SSH

```bash
exit
```

---

## 7. File Management & Process Control on EC2

### Check file modification/creation time

```bash
ls -l <filename>         # Shows last modification time
stat <filename>          # Shows access, modify, and change times
```

### Kill and restart Gunicorn Flask app

```bash
ps aux | grep gunicorn   # Find Gunicorn process
kill <PID>               # Kill by PID
pkill gunicorn           # Kill all Gunicorn processes
```

### Start Gunicorn as a daemon (background)

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --preload --daemon
```

---

## Notes

- Use the correct activation command for your OS.
- Always install dependencies inside the active venv.
- For model updates, see REPLACING_MODEL.md for SCP instructions.
- For production, use Gunicorn or another WSGI server instead of Flask's built-in server.
- Open port 5000 in your EC2 security group to allow external access.
