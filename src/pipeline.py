import os;os.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True";import gc,copy;from PIL.Image import Image;import threading as th,time as t;from pipelines.models import TextToImageRequest as T2IR;import torch as tc;from torchao.quantization import int8_weight_only as iw,quantize_ as q_;from torch import Generator as G;from diffusers import FluxPipeline as FP,FluxTransformer2DModel as FT2D,AutoencoderKL as AKL;from diffusers.image_processor import VaeImageProcessor as VIP;from transformers import T5EncoderModel as T5E,T5TokenizerFast as T5T,CLIPTokenizer as CT,CLIPTextModel as CTM;c="black-forest-labs/FLUX.1-schnell";x=lambda:gc.collect();y=lambda d,n:f"{n}"in d and d.pop(n).join();z=lambda s:not s or len(s)==0
def f(a):
 if z(a.get("text_encoder")):a["text_encoder"]=CTM.from_pretrained(c,subfolder="text_encoder",low_cpu_mem_usage=1,torch_dtype=tc.bfloat16)
 if z(a.get("text_encoder_2")):a["text_encoder_2"]=T5E.from_pretrained(c,subfolder="text_encoder_2",low_cpu_mem_usage=1,torch_dtype=tc.bfloat16)
 if z(a.get("tokenizer")):a["tokenizer"]=CT.from_pretrained(c,subfolder="tokenizer")
 if z(a.get("tokenizer_2")):a["tokenizer_2"]=T5T.from_pretrained(c,subfolder="tokenizer_2")
def g(a):
 if z(a.get("transformer")):
  a["transformer"]=FT2D.from_pretrained("/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/transformer",low_cpu_mem_usage=1,torch_dtype=tc.bfloat16)
  q_(a["transformer"],iw(),device="cuda")
def h(a):
 if z(a.get("vae")):a["vae"]=AKL.from_pretrained(c,subfolder="vae",torch_dtype=tc.bfloat16).to("cuda")
 if z(a.get("vae_scale_factor")) and not z(a.get("vae")):a["vae_scale_factor"]=2**len(a["vae"].config.block_out_channels)
 if z(a.get("image_processor")) and not z(a.get("vae_scale_factor")):a["image_processor"]=VIP(vae_scale_factor=a["vae_scale_factor"])
def i(a):f(a);h(a);return a
def j(b,a):
 p=FP.from_pretrained(c,text_encoder=a["text_encoder"],text_encoder_2=a["text_encoder_2"],tokenizer=a["tokenizer"],tokenizer_2=a["tokenizer_2"],transformer=None,vae=None)
 s=t.time();p.to('cuda')
 with tc.no_grad():
  pe,ppe,ti=p.encode_prompt(prompt=b,prompt_2=None,max_sequence_length=256)
 a.update({"text_encoder":None,"text_encoder_2":None,"tokenizer":None,"tokenizer_2":None})
 return pe,ppe,ti
def k(pe,ppe,w,h,s,a):
 p=FP.from_pretrained(c,transformer=a["transformer"],text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,vae=None)
 g=G(p.device).manual_seed(s) if s else None
 l=p(prompt_embeds=pe,pooled_prompt_embeds=ppe,num_inference_steps=4,guidance_scale=0.0,width=w,height=h,generator=g,output_type="latent").images
 a["transformer"]=None
 return l
def m(l,w,h,a):
 v=a["vae"].to("cuda");vsf=a["vae_scale_factor"];ip=a["image_processor"]
 h=h or 64*vsf;w=w or 64*vsf
 with tc.no_grad():
  l=FP._unpack_latents(l,h,w,vsf)
  l=(l/v.config.scaling_factor)+v.config.shift_factor
  img=v.decode(l,return_dict=False)[0]
  img=ip.postprocess(img,output_type="pil")
 return img
def n(r,p):
 a,l=p;x();y(l,"load_threadtext_encoder")
 pe,ppe,ti=j(r.prompt,a);x();g(a)
 lt=th.Thread(target=f,args=(a,));l["load_threadtext_encoder"]=lt;lt.start();x()
 imgs=m(k(pe,ppe,r.width,r.height,r.seed,a),r.width,r.height,a)
 return imgs[0]
def o():
 a={};l={};a=i(a)
 for _ in range(3):
  n(T2IR(prompt=""),(a,l))
 return a,l
