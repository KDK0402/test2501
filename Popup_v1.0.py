import os
import time
import threading
import queue
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import tkinter as tk
from tkinter import messagebox

class ChatFileHandler(FileSystemEventHandler):
    def __init__(self, file_path, keywords):
        self.file_path = file_path
        self.keywords = keywords
        self.last_position = 0
        self.popup_count = 0
        self.message_queue = queue.Queue()
        self.root = None
        
        # GUI 초기화
        self.initialize_gui()
        
        # 파일 위치 초기화
        self.initialize_position()
        
    def initialize_gui(self):
        # GUI 스레드 생성
        self.root = tk.Tk()
        self.root.withdraw()  # 메인 창 숨기기
        
        # GUI 업데이트 시작
        self.process_message_queue()
        
    def process_message_queue(self):
        try:
            # 큐에서 메시지 확인
            while not self.message_queue.empty():
                message = self.message_queue.get_nowait()
                self.create_popup(message)
                
            # 100ms마다 큐 확인
            self.root.after(100, self.process_message_queue)
            
        except Exception as e:
            print(f"메시지 처리 오류: {e}")
            self.root.after(100, self.process_message_queue)

    def create_popup(self, message):
        try:
            popup = tk.Toplevel(self.root)
            popup.wm_title("키워드 알림")
            
            # 팝업 크기 설정
            popup_width = 400
            popup_height = 150
            
            # 화면 크기 가져오기
            screen_width = self.root.winfo_screenwidth()
            screen_height = self.root.winfo_screenheight()
            
            # 우측 하단 기준 위치 계산
            base_x = screen_width - popup_width - 50
            base_y = screen_height - popup_height - 50
            
            # 팝업 오프셋 계산 (위로 쌓이도록)
            offset = 30 * self.popup_count
            x = base_x
            y = base_y - offset
            
            # 화면 상단 경계 체크
            if y < 0:
                y = base_y
                self.popup_count = 0
            
            # 팝업 위치와 크기 설정
            popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
            
            # 항상 최상위에 표시
            popup.lift()
            popup.attributes('-topmost', True)
            
            # 메시지 표시
            label = tk.Label(
                popup, 
                text=message, 
                wraplength=350, 
                pady=20,
                justify='center',
                font=('Arial', 11, 'bold')
            )
            label.pack(expand=True)
            
            # 팝업 카운트 증가
            self.popup_count += 1
            
            # 20초 후 자동으로 닫기 (10000 -> 20000 밀리초로 변경)
            popup.after(20000, lambda: self.close_popup(popup))
            
        except Exception as e:
            print(f"팝업 생성 오류: {e}")

    def close_popup(self, popup):
        try:
            self.popup_count -= 1
            popup.destroy()
        except:
            pass

    def initialize_position(self):
        if os.path.exists(self.file_path):
            with open(self.file_path, 'r', encoding='cp949') as file:
                file.seek(0, 2)  # 파일의 끝으로 이동
                self.last_position = file.tell()

    def on_modified(self, event):
        if event.src_path == self.file_path:
            self.check_new_content()

    def check_new_content(self):
        try:
            with open(self.file_path, 'r', encoding='cp949') as file:
                file.seek(self.last_position)
                new_content = file.read()
                self.last_position = file.tell()

                if new_content:
                    lines = new_content.splitlines()
                    for line in lines:
                        for keyword in self.keywords:
                            if keyword in line:
                                # 메시지를 라인 내용만으로 단순화
                                print(line)  # 터미널에 라인 내용만 출력
                                self.message_queue.put(line)  # 팝업에도 라인 내용만 전달
        except Exception as e:
            print(f"파일 읽기 오류: {e}")

def start_monitoring(file_path, keywords):
    event_handler = ChatFileHandler(file_path, keywords)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(file_path), recursive=False)
    observer.start()

    try:
        event_handler.root.mainloop()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # 파일 경로 설정
    file_path = r"C:\Users\infomax\Documents\K-Bond Messenger Chat\채권_블커본드_20250108_080744.txt"
    keywords = ["25.9.3", "23-8", "22-13", "15-8", "26.1.2", "26.3.3", "6-1", "21-1", "24-3"]
    
    # 파일 존재 여부 확인
    if not os.path.exists(file_path):
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
        exit(1)
        
    print(f"모니터링할 파일: {file_path}")
    print(f"검색할 키워드: {keywords}")
    print("파일 모니터링을 시작합니다...")
    
    try:
        start_monitoring(file_path, keywords)
    except Exception as e:
        print(f"모니터링 중 오류 발생: {e}")


# 151 line에 보기 위한 키워드를 수기로 입력해서 해당 파일 이용