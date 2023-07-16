

S = 1,2,3,4,5,6,7,8,9
SP = 700,800,900,1000,1100
BP = 800,900,1000,1100,1200

for s in S:
 for sp in SP:
  for bp in BP:
    if sp>=bp:
      continue
    else:
      layout = pya.Layout()
      small_count = s
      small_pitch = sp
      big_count = 10-small_count
      big_pitch = bp
      waveguide_start = -40000
      waveguide_end = 0
      thickness_start = 0
      thickness_end = 400
      extra_length = 10000
      name = "big"+str(big_pitch)+"-"+str(big_count)+"_"+"small"+str(small_pitch)+"-"+str(small_count)
      
      wg_x1 = waveguide_start
      wg_x2 = waveguide_end
      wg_y1 = thickness_start
      wg_y2 = thickness_end
      
      first_X1 = waveguide_end + (big_pitch/2)
      x1_big = []
      x1_big.append(first_X1)
      for i in range(big_count-1):
          first_X1 += big_pitch
          x1_big.append(first_X1)
      x2_big = []
      for i in x1_big:
          x2_big.append(i+(big_pitch/2))
      
      first_x1 = x2_big[-1] + (small_pitch/2)
      x1_small = []
      x1_small.append(first_x1)
      for i in range(small_count-1):
          first_x1 += small_pitch
          x1_small.append(first_x1)
      x2_small = []
      for i in x1_small:
          x2_small.append(i+(small_pitch/2))
      
      x1 = x1_big +x1_small
      x2 = x2_big + x2_small
      y1 = [thickness_start] * (small_count + big_count)
      y2 = [thickness_end] * (small_count + big_count)
      
      extra_x1 = x2[-1] + (small_pitch/2)
      extra_x2 = extra_x1+extra_length
      extra_y1 = thickness_start
      extra_y2 = thickness_end
      
      
      
      top_cell = layout.create_cell(name)
      
      
      # Create a rectangle
      layer = layout.layer(1, 0)  # Layer 1, datatype 0
      rect_wg = pya.Box(wg_x1, wg_y1, wg_x2, wg_y2)  # Rectangle coordinates (in nm)
      top_cell.shapes(layer).insert(rect_wg)
      
      for xi1, yi1, xi2, yi2 in zip(x1, y1, x2, y2):
        rect_grating = pya.Box(xi1, yi1, xi2, yi2)  # Rectangle coordinates (in nm)
        top_cell.shapes(layer).insert(rect_grating)
      rect_extra = pya.Box(extra_x1, extra_y1, extra_x2, extra_y2)
      top_cell.shapes(layer).insert(rect_extra)   
      
      layout.write("C:\\Users\\limyu\\Downloads\\"+name+".gds")