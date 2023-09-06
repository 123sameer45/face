import cv2
import mediapipe as mp

import tempfile
import streamlit as st


#demo video 
DEMO_VIDEO = '_109__import_FPpreview.mp4'



#mediapipe inbuilt solutions 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils





def main():

    #title 
    st.title('Face Detection App')

    #sidebar title
    st.sidebar.title('Face Detection App')

    st.sidebar.subheader('Parameters')
    #creating a button for webcam
    use_webcam = st.sidebar.button('Use Webcam')
    #creating a slider for detection confidence 
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value =0.0,max_value = 1.0,value = 0.5)
    
    #model selection 
    model_selection = st.sidebar.selectbox('Model Selection',options=[0,1,2])
    st.markdown(' ## Output')
    stframe = st.empty()
    
    
    #temporary file name 
    tfflie = tempfile.NamedTemporaryFile(delete=False)

   

    if use_webcam:
            vid = cv2.VideoCapture(0)
    else:
            vid = cv2.VideoCapture(DEMO_VIDEO)
            tfflie.name = DEMO_VIDEO
    
    
    #values 
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    #codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    codec = cv2.VideoWriter_fourcc('V','P','0','9')
    out = cv2.VideoWriter('output1.webm', codec, fps, (width, height))


    # drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


    with mp_face_detection.FaceDetection(
    model_selection=model_selection, min_detection_confidence=detection_confidence) as face_detection:
        
        while vid.isOpened():

            ret, image = vid.read()

            if not ret:
                break
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
            stframe.image(image,use_column_width=True)

        vid.release()
        out.release()
        cv2.destroyAllWindows()

    st.success('Video is Processed')
    st.stop()

if __name__ == '__main__':
    main()
