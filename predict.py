from PIL import Image

from arcface import Arcface

if __name__ == "__main__":
    model = Arcface()
        
    #----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'fps'表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #----------------------------------------------------------------------------------------------------------#
    mode            = "predict"
    #-------------------------------------------------------------------------#
    #   test_interval   用于指定测量fps的时候，图片检测的次数
    #                   理论上test_interval越大，fps越准确。
    #   fps_test_image  fps测试图片
    #-------------------------------------------------------------------------#
    test_interval   = 100
    fps_test_image  = 'img/1_001.jpg'
    
    if mode == "predict":
        while True:
            image_1 = input('Input image_1 filename:')
            try:
                image_1 = Image.open(image_1)
            except:
                print('Image_1 Open Error! Try again!')
                continue

            image_2 = input('Input image_2 filename:')
            try:
                image_2 = Image.open(image_2)
            except:
                print('Image_2 Open Error! Try again!')
                continue
            
            probability = model.detect_image(image_1,image_2)
            print(probability)

    elif mode == "fps":
        img = Image.open(fps_test_image)
        tact_time = model.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')