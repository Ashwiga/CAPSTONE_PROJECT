# AI-Enabled Student Safety and Attendance Monitoring System

## 📌 Project Overview

The **AI-Enabled Student Safety and Attendance Monitoring System** is a real-time intelligent system designed to monitor student boarding and deboarding in school buses using face detection, face recognition, GPS tracking, and automated notifications.

The system uses computer vision and AI technologies to automatically detect and recognize students, record attendance, capture location details, and send email and SMS alerts to parents and school authorities. This helps improve student safety, transparency, and monitoring efficiency.

---

## 🚀 Features

* Real-time face detection using YOLO
* Face recognition using FaceNet
* Automatic student boarding and deboarding attendance
* GPS location tracking
* Email notifications to parents
* SMS alerts using Twilio
* PostgreSQL database storage
* Flask web dashboard for monitoring
* Secure and scalable system
* Real-time student tracking

---

## 🛠️ Technologies Used

### Programming Language

* Python 3.10+

### Framework

* Flask

### Database

* PostgreSQL

### AI & Computer Vision

* OpenCV
* YOLOv8
* FaceNet
* Cosine Similarity

### Notification Services

* SMTP Email Service
* Twilio API

### Frontend

* HTML
* CSS
* JavaScript

### Other Tools

* Git & GitHub
* VS Code



## ⚙️ Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/student-safety-system.git
cd student-safety-system
```

---

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

Activate environment:

**Windows**

```bash
venv\Scripts\activate
```

**Mac/Linux**

```bash
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Setup PostgreSQL Database

Create database:

```
student_safety_db
```

Create tables:

```
students
attendance_logs
```

Update database credentials in:

```
app.py
```

---

### Step 5: Add Twilio and Email Configuration

Update in:

```
sms_alert.py
email_alert.py
```

Add:

* Twilio Account SID
* Auth Token
* Phone Number
* Email SMTP credentials

---

### Step 6: Run the Application

```bash
python app.py
```

---

## 🌐 Access the System

Open browser:

```
http://127.0.0.1:5000
```

Dashboard will display:

* Student Name
* Boarding Status
* Deboarding Status
* Location
* Date & Time

---

## 📊 System Workflow

1. Camera captures video
2. YOLO detects faces
3. FaceNet recognizes students
4. Attendance recorded
5. GPS location captured
6. Email sent to parents
7. SMS alert sent
8. Data stored in PostgreSQL
9. Flask dashboard displays logs

---

## 📧 Email Alert Example

```
Student Name: Ashwiga
Status: Boarded
Location: 11.0168, 76.9558
Time: 08:45 AM
Date: 12-03-2026
```

---

## 📱 SMS Alert Example

```
Student Ashwiga has boarded the bus.
Location: 11.0168, 76.9558
Time: 08:45 AM
```

---

## ✅ Advantages

* Real-time monitoring
* High accuracy face recognition
* Automatic attendance
* GPS tracking
* Instant parent notification
* Secure database storage
* Easy dashboard monitoring
* Scalable system

---

## 🎯 Future Enhancements

* Mobile application integration
* Live bus tracking on map
* Multi-bus support
* Cloud deployment
* AI-based anomaly detection
* Student ID verification

---


## 📜 License

This project is developed for academic and research purposes.

---

## ⭐ Support

If you like this project, give it a ⭐ on GitHub.
