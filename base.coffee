#!/usr/bin/env iced
require 'fy'
nooocl = require 'nooocl'
{
  CLBuffer
  CLHost
  CLContext
  CLCommandQueue
  NDRange
} = nooocl
crypto = require 'crypto'
fs = require 'fs'
{PNG} = require 'pngjs'

####################################################################################################
# config
####################################################################################################
image_size_x = 1920*2
image_size_y = 1080*2

tex_size_x = 300
tex_size_y = 300
tex_count = 1
####################################################################################################
# gpu
####################################################################################################

host = CLHost.createV11()
{defs} = host.cl

gpu_list = []
platform_list = host.getPlatforms()
if !platform_list.length
  throw new Error "missing compatible opencl plaftorm"

for platform in platform_list
  gpu_list.append platform.gpuDevices()
if !gpu_list.length
  throw new Error "missing compatible opencl gpu "

p "gpu count: #{gpu_list.length}"
gpu = gpu_list[0];
p "device: #{gpu.name} #{gpu.platform.name}"
ctx = new CLContext gpu

queue = new CLCommandQueue ctx, gpu
####################################################################################################
# buffers
####################################################################################################

rect_list_buf_size = 1000*8*4
rect_list_buf_host = Buffer.alloc rect_list_buf_size
rect_list_buf_gpu  = new CLBuffer ctx, defs.CL_MEM_READ_ONLY, rect_list_buf_size

image_size_byte = image_size_x*image_size_y*4
image_buf_host = Buffer.alloc image_size_byte
#image_buf_gpu  = new CLBuffer ctx, defs.CL_MEM_READ_WRITE, image_size_byte
image_buf_gpu  = new CLBuffer ctx, defs.CL_MEM_WRITE_ONLY, image_size_byte

png_data = fs.readFileSync './tex/index.png'
png = PNG.sync.read png_data
tex_size_bytes = tex_size_x*tex_size_y*4*tex_count
image_atlas_buf_gpu  = new CLBuffer ctx, defs.CL_MEM_WRITE_ONLY, tex_size_bytes



# TODO tex atlas
# TODO move tex atlas to gpu
####################################################################################################
# kernel
####################################################################################################

program = ctx.createProgram fs.readFileSync "./kernel.cl", 'utf-8'
await program.build('').then defer()
build_status = program.getBuildStatus gpu
if build_status < 0
  build_error = program.getBuildLog gpu
  throw new Error "can't build. reason: #{build_error}"
kernel_draw_call_rect_list = program.createKernel "draw_call_rect_list"
kernel_global_size = new NDRange image_size_x*image_size_y
kernel_local_size  = new NDRange 32


####################################################################################################
# hash
####################################################################################################

hash = (msg_buf, cb)->
  # TODO lock
  scene_seed = crypto.createHash('sha256').update(msg_buf).digest()
  
  offset = 0
  rect_list = []
  for i in [0 ... 4]
    rect_list.push {
      x : scene_seed[offset++ % scene_seed.length]
      y : scene_seed[offset++ % scene_seed.length]
      w : scene_seed[offset++ % scene_seed.length]
      h : scene_seed[offset++ % scene_seed.length]
      t : scene_seed[offset++ % scene_seed.length] % tex_count
    }
  
  for rect,idx in rect_list
    rect_offset = idx*8*4
    rect_list_buf_host.writeInt32BE(rect.x, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32BE(rect.y, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32BE(rect.w, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32BE(rect.h, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32BE(rect.t, rect_offset); rect_offset += 4
  
  queue.enqueueWriteBuffer rect_list_buf_gpu, 0, rect_list_buf_size, rect_list_buf_host

  kernel_draw_call_rect_list.setArg 0, rect_list_buf_gpu
  #kernel_draw_call_rect_list.setArg 1, image_atlas
  kernel_draw_call_rect_list.setArg 2, image_buf_gpu
  kernel_draw_call_rect_list.setArg 3, rect_list.length
  kernel_draw_call_rect_list.setArg 4, image_size_x
  kernel_draw_call_rect_list.setArg 5, tex_size_x

  queue.enqueueNDRangeKernel kernel_draw_call_rect_list, kernel_global_size, kernel_local_size

  # call kernel
  # move image to host
  cb null, crypto.createHash('sha256').update(image_buf_host).digest()
####################################################################################################
# test
####################################################################################################

msg = Buffer.alloc 80
for i in [0 ... 80]
  msg[0] = i;
await hash "", defer(err, hash); throw err if err

p hash