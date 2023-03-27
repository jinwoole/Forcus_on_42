# Forcus on 42
Developed by jinwoole, [juha](https://github.com/contemplation-person), [seonhoki02](https://github.com/kosuha)  
42서울 2023 코알리숑 해커톤에서 제작되었고, 감 코알리숑의 우승작입니다.  
이 프로젝트는 거북목을 예방하고, 화면 잠금을 잊어버림으로 인한 보안 문제를 해결하기 위해 개발되었습니다.

## 설치 방법

1. 이 프로젝트를 클론 또는 다운로드하세요.  
2. 프로젝트 폴더로 이동한 후, 다음 명령어를 실행하여 필요한 라이브러리를 설치하세요:

```sh
pip install opencv-python
pip install face-recognition
pip install pystray
```
## 실행 방법

설치가 완료되면, 프로젝트 폴더에서 다음 명령어를 실행하여 프로그램을 실행할 수 있습니다:

```sh
python runner.py
```
## 사용 방법

프로그램 실행 후 트레이 아이콘을 통해 프로그램을 제어할 수 있습니다.  

Lock: 화면 잠금 기능을 활성화 또는 비활성화합니다.  
Turtle: 거북목 예방 기능을 활성화 또는 비활성화합니다.  
Start: 프로그램을 시작합니다. Lock과 Turtle 중 하나 이상을 선택해야 합니다.  
Stop: 프로그램 기능을 중지합니다.  
Exit: 프로그램을 종료합니다.  
프로그램을 종료하면, 프로그램이 생성한 인식 파일은 자동으로 삭제됩니다.

## 주의사항

프로그램을 사용하기 전에 카메라를 통해 사용자의 얼굴이 잘 인식되는지 확인하시기 바랍니다. 얼굴 인식이 제대로 되지 않으면 프로그램의 기능이 정확하게 작동하지 않을 수 있습니다.  
사용을 위해선 라이브러리 설치에 상당한 시간이 소요됩니다.   
if error occurs, use python3 (and change the runner.py code to python -> python3).  
 
