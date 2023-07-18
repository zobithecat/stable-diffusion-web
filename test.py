from diffusers import StableDiffusionPipeline
import torch
from flask import Flask, request
import time

app = Flask(__name__, 
            static_url_path="")

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

def stDiffusionRunner(prompt):
   image = pipe(prompt).images[0]
   imagePath = "/images/result"+str(int(time.time()))+".jpg"
   filePath = "./static" + imagePath
   image.save(filePath)
   return imagePath

@app.route('/image')
def requestImage():
   ns = request.args.to_dict()
   prompt = ns['requestText']
   print("prompt:", prompt)
   imagePath = stDiffusionRunner(prompt)
   return imagePath

@app.route('/')
def root():
    return app.send_static_file('index.html')

if __name__ == '__main__':
  app.run(debug=True, host="0.0.0.0")