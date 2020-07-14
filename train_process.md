<Network Architecture>
  model : Yolo v3 - Darknet 53
  Input shape : (416, 416, 3)

  <speed>
    yolo model : 0.46 fps
    post processing : 33 fps

===Train 1===
  Start time : 2020.06.19 21:14:29
  End time : 2020.06.20 21:09
  DB : WFLW
  DB_path : 'C:\Users\th_k9\Desktop\Yolov3withFacelandmark\annotation_preparation\WFLW\annotations'
  DB_num : 7500
  last_loss : 1.1842
  test : x
  Reason to stop training : 가족여행 전에 300VW로 돌려 놓으려고
    setting = {
      BATCH_SIZE : 8,
      EPOCH : 10000,
      data_dir : './annotation_preparation/WFLW',
      ckpt_dir : './checkpoints',
      num_landmarks : 196,
      lr_rate : 0.00001}


===Train 2===
  Start time : 2020.06.22
  End time : 2020.06.23 13:30:00
  DB : 300VW
  DB_path : 'E:/DB_FaceLandmark/300VW_frame'
  DB_num : 2,223 (중복 다수)
  last_loss : 0.0031
  test : 망함
  Reason to stop training : 더이상 loss가 안내려감 (0.0000001)
    setting = {
      BATCHO_SIZE : 8,
      EPCH : 10000,
      data_dir : './annotation_preparation/300VW_train'
      ckpt_dir : 'E:/checkpoints'
      num_landmarks : 136,
      lr_rate : 0.00001, 0.000001, 0.0000001
    }
  fps : 4
  모델 수행 속도와 정확도 높여야함
