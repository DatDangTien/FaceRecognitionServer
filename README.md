## **Quick Start**

This project uses FaceNet and MTCNN (based on [facenet-pytorch](https://github.com/timesler/facenet-pytorch)).
For more details, see the article: https://viblo.asia/p/nhan-dien-khuon-mat-voi-mang-mtcnn-va-facenet-phan-2-bJzKmrVXZ9N

1. Install

   ```bash
   # Clone Repo
   git clone https://github.com/pewdspie24/FaceNet-Infer.git
   cd FaceNet-Infer

   # Install dependencies
   pip install -r requirements.txt
   ```

2. Register & manage users (run first)

   Use this step to register new users and manage the user list. Embeddings and usernames are saved to the `data/` folder (`faceslist.pth` or `faceslistCPU.pth`, and `usernames.npy`).

   ```bash
   # User registration & management menu
   python face_capture_noSave.py
   ```

   In the menu you can:

   - Register a new user (capture multiple high‑quality samples)
   - View the user list
   - Delete a user

3. Run face recognition

   ```bash
   python face_recognition.py
   ```

4. Optional utilities

   - `face_detect.py`: basic face detection preview
   - `update_faces.py`: legacy script to rebuild face lists (usually not needed if you use `face_capture_noSave.py`)

### Configuration

All thresholds and parameters (quality checks, recognition threshold, MTCNN, tracking, camera, display, registration settings) are configurable in `face_config.py`. Adjust these values to fit your environment:

```bash
# Edit config
vim face_config.py
```
