__kernel void draw_call_rect_list(
	__global int *rect_list,
	__global int4 *image_atlas,
	__global int4 *image_result,
	const unsigned int rect_list_length,
	const unsigned int size_x,
	const unsigned int tex_size_x
	)
{
	// per pixel shader
	int id = get_global_id(0);
	int x = id % size_x;
	int y = id / size_x;
	
	int i;
	int rect_id = -1;
	for(i = 0;i < rect_list_length;i++){
		int offset = i*8;
		int rect_x = rect_list[offset  ];
		int rect_y = rect_list[offset+1];
		int rect_w = rect_list[offset+2];
		int rect_h = rect_list[offset+3];
		
		bool fit_x = x >= rect_x && x < rect_x + rect_w;
		bool fit_y = y >= rect_y && y < rect_y + rect_h;
		rect_id = (fit_x && fit_y) ? i : rect_id;
	}
	
	if (rect_id == -1) {
		image_result[id].x = 0;
		image_result[id].y = 0;
		image_result[id].z = 0;
		image_result[id].w = 0;
		return;
	}
	int rect_x = rect_list[rect_id  ];
	int rect_y = rect_list[rect_id+1];
	int rect_tex_idx = rect_list[rect_id+4];
	
	int tex_offset_x = x - rect_x;
	int tex_offset_y = y - rect_y;
	int tex_offset = tex_offset_x + tex_offset_y*tex_size_x;
	
	image_result[id] = image_atlas[tex_offset];
}
