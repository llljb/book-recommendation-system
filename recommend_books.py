import numpy as np
import tensorflow as tf

# 加载模型
model = tf.keras.models.load_model('（1）new_book_recommendation_model.h5')

def recommend(user_id, book_id):
    user_id = np.array([user_id])
    book_id = np.array([book_id])
    prediction = model.predict([user_id, book_id])
    return prediction[0][0]

if __name__ == "__main__":
    user_id = int(input("请输入用户ID: "))
    book_id = int(input("请输入图书ID: "))
    rating = recommend(user_id, book_id)
    print(f"推荐评分: {rating}")
