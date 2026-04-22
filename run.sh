#!bin/bash
echo "<<<<<<<<<<<< ASL DETECTION >>>>>>>>>>>"
echo "-------- ResNet50 x MediaPipe --------"
echo "                                      "
cd frontend
echo "starting the main api"
start api.html

cd ../backend
echo "--> activating virtual environment"
source venv312/Scripts/activate
echo "--> launching main backend file"
uvicorn main:app
echo "                    "
echo "ASL Detector closed "
