import os as o;o.environ['PYTORCH_CUDA_ALLOC_CONF']="expandable_segments:True"
import gc as g
import copy as c
from PIL.Image import Image as I
import threading as t
from concurrent.futures import ThreadPoolExecutor as TPE
from pipelines.models import TextToImageRequest as TIR
import torch as tc
from torch import Generator as G
from diffusers import FluxPipeline as FP, FluxTransformer2DModel as FTM, AutoencoderKL as AKL
from diffusers.image_processor import VaeImageProcessor as VIP
from transformers import T5EncoderModel as T5EM, T5TokenizerFast as T5TF, CLIPTokenizer as CT, CLIPTextModel as CTM
import transformers as tr;tr.utils.logging.disable_progress_bar()
import diffusers as df;df.utils.logging.disable_progress_bar()
C="black-forest-labs/FLUX.1-schnell";cps={};lts={}
def ec():g.collect()
def f1(cps):
    if cps.get("text_encoder")is None:cps["text_encoder"]=CTM.from_pretrained(C,subfolder="text_encoder",low_cpu_mem_usage=True,torch_dtype=tc.bfloat16)
    if cps.get("text_encoder_2")is None:cps["text_encoder_2"]=T5EM.from_pretrained(C,subfolder="text_encoder_2",low_cpu_mem_usage=True,torch_dtype=tc.bfloat16)
    if cps.get("tokenizer")is None:cps["tokenizer"]=CT.from_pretrained(C,subfolder="tokenizer")
    if cps.get("tokenizer_2")is None:cps["tokenizer_2"]=T5TF.from_pretrained(C,subfolder="tokenizer_2")
def f2(cps):
    if cps.get("transformer")is None:cps["transformer"]=FTM.from_pretrained("/root/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-schnell/snapshots/741f7c3ce8b383c54771c7003378a50191e9efe9/transformer",low_cpu_mem_usage=True,torch_dtype=tc.bfloat16)
def f3(cps):
    if cps.get("vae")is None:cps["vae"]=AKL.from_pretrained(C,subfolder="vae",torch_dtype=tc.bfloat16).to("cuda")
    if cps.get("vae_scale_factor")is None and cps.get("vae")is not None:cps["vae_scale_factor"]=2**len(cps["vae"].config.block_out_channels)
    if cps.get("image_processor")is None and cps.get("vae_scale_factor")is not None:cps["image_processor"]=VIP(vae_scale_factor=cps["vae_scale_factor"])
def f4(cps):
    with TPE()as e:
        fs=[e.submit(f1,cps),e.submit(f2,cps),e.submit(f3,cps)]
        [fu.result()for fu in fs]
    return cps
def f5(p:str,cps):
    pl=FP.from_pretrained(C,text_encoder=cps["text_encoder"],text_encoder_2=cps["text_encoder_2"],tokenizer=cps["tokenizer"],tokenizer_2=cps["tokenizer_2"],transformer=None,vae=None).to("cuda")
    with tc.no_grad():
        pe,ppe,ti=pl.encode_prompt(prompt=p,prompt_2=None,max_sequence_length=256)
    cps["text_encoder"]=None;cps["text_encoder_2"]=None;cps["tokenizer"]=None;cps["tokenizer_2"]=None
    return pe,ppe,ti
def f6(pe,ppe,w,h,s,cps):
    pl=FP.from_pretrained(C,transformer=cps["transformer"],text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None,vae=None).to("cuda")
    gen=G(pl.device).manual_seed(s)if s is not None else None
    l=pl(prompt_embeds=pe,pooled_prompt_embeds=ppe,num_inference_steps=4,guidance_scale=0.0,width=w,height=h,generator=gen,output_type="latent").images
    cps["transformer"]=None
    return l
def f7(l,w,h,cps):
    v=cps["vae"].to("cuda");vsf=cps["vae_scale_factor"];ip=cps["image_processor"]
    with tc.no_grad():
        l=FP._unpack_latents(l,h,w,vsf);l=(l/v.config.scaling_factor)+v.config.shift_factor
        im=v.decode(l,return_dict=False)[0];im=ip.postprocess(im,output_type="pil")
    return im
def n(rq,pl)->I:
    cps,lts=pl;ec()
    if"load_threadtext_encoder"in lts:lts.pop("load_threadtext_encoder").join()
    pe,ppe,ti=f5(rq.prompt,cps)
    ltte=t.Thread(target=f1,args=(cps,));lts["load_threadtext_encoder"]=ltte;ltte.start()
    ec()
    if"load_thread_transformer"in lts:lts.pop("load_thread_transformer").join()
    l=f6(pe,ppe,rq.width,rq.height,rq.seed,cps)
    ltt=t.Thread(target=f2,args=(cps,));lts["load_thread_transformer"]=ltt;ltt.start()
    ec()
    ims=f7(l,1024,1024,cps);return ims[0]
def o():
    cps={};lts={};cps=f4(cps);[n(TIR(prompt=""),(cps,lts))for _ in range(3)]
    return(cps,lts)
