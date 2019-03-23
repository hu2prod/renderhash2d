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

file_list = fs.readdirSync('./tex')
tex_count = file_list.length
tex_size_bytes = tex_size_x*tex_size_y*4*tex_count

tex_buf_host = Buffer.alloc tex_size_x*tex_size_y*4
image_atlas_buf_gpu  = new CLBuffer ctx, defs.CL_MEM_WRITE_ONLY, tex_size_bytes

tex_offset = 0
for file in file_list
  p file
  full_file = "./tex/#{file}"
  png_data = PNG.sync.read fs.readFileSync full_file
  {
    data
    width
    height
  } = png_data
  for x in [0 ... tex_size_x]
    for y in [0 ... tex_size_y]
      src_offset = 4*(x + y*width)
      dst_offset = 4*(x + y*tex_size_x)
      tex_buf_host[dst_offset+0] = data[src_offset+0]
      tex_buf_host[dst_offset+1] = data[src_offset+1]
      tex_buf_host[dst_offset+2] = data[src_offset+2]
      tex_buf_host[dst_offset+3] = 255

  await queue.waitable().enqueueWriteBuffer(image_atlas_buf_gpu, tex_offset, tex_buf_host.length, tex_buf_host).promise.then defer()
  tex_offset += tex_buf_host.length

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
  # debug
  rect_list = [
    {
      x:10
      y:20
      w:300
      h:300
      t:0
    }
  ] 
  for rect,idx in rect_list
    rect_offset = idx*8*4
    rect_list_buf_host.writeInt32LE(rect.x, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32LE(rect.y, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32LE(rect.w, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32LE(rect.h, rect_offset); rect_offset += 4
    rect_list_buf_host.writeInt32LE(rect.t, rect_offset); rect_offset += 4
  
  queue.enqueueWriteBuffer rect_list_buf_gpu, 0, rect_list_buf_size, rect_list_buf_host

  kernel_draw_call_rect_list.setArg 0, rect_list_buf_gpu
  kernel_draw_call_rect_list.setArg 1, image_atlas_buf_gpu
  kernel_draw_call_rect_list.setArg 2, image_buf_gpu
  kernel_draw_call_rect_list.setArg 3, rect_list.length, "uint"
  kernel_draw_call_rect_list.setArg 4, image_size_x, "uint"
  kernel_draw_call_rect_list.setArg 5, tex_size_x, "uint"

  queue.enqueueNDRangeKernel kernel_draw_call_rect_list, kernel_global_size, kernel_local_size

  queue.waitable().enqueueReadBuffer(image_buf_gpu, 0, image_size_byte, image_buf_host).promise.then defer()
  # move image to host
  p image_buf_host

  options = { colorType: 6 }; # RGBA
  buffer = PNG.sync.write({data:image_buf_host, width:1920*2, height:1080*2}, options);
  fs.writeFileSync('result.png', buffer);

  cb null, crypto.createHash('sha256').update(image_buf_host).digest()
####################################################################################################
# test
####################################################################################################

msg = Buffer.alloc 80
for i in [0 ... 80]
  msg[0] = i;
await hash "", defer(err, hash); throw err if err


p hash