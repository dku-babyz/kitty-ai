# tmp.py  – 1회만 로드, 이후 빠른 테스트용
from inference.text import TextPredictor
from inference.image import ImagePredictor
from inference.dictionary import DictionaryChecker
import time, json

checker  = DictionaryChecker("inference/dictionary/사전.csv")
txt_pred = TextPredictor("inference/text/model")
img_pred = ImagePredictor("inference/image/model_ts.pt")

def bench():

    s=time.time(); print("warmup : ", checker("야 개삽년이! 시발 장난해!?")); print("dict", (time.time()-s)*1e3,"ms")

    s=time.time(); print("warmup : " ,txt_pred("유해 텍스트"));            print("text", (time.time()-s)*1e3,"ms")

    s=time.time(); print("warmup : ",img_pred("example.jpg")[0]);         print("img ", (time.time()-s)*1e3,"ms")



    s=time.time(); print(checker("야 개삽년이! 시발 장난해!?")); print("dict", (time.time()-s)*1e3,"ms")

    s=time.time(); print(txt_pred("유해 텍스트"));            print("text", (time.time()-s)*1e3,"ms")

    s=time.time(); print(img_pred("example.jpg")[0]);         print("img ", (time.time()-s)*1e3,"ms")

if __name__=="__main__":
    bench()
