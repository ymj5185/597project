from towhee import pipe, ops, DataCollection

if __name__ == '__main__':

    p = (
        pipe.input('url')
        .map('url', 'img', ops.image_decode.cv2_rgb())
        .map('img', 'text', ops.image_captioning.expansionnet_v2(model_name='expansionnet_rf'))
        .output('img', 'text')
    )

    DataCollection(p('./demo_material/micheal.jpg')).show()