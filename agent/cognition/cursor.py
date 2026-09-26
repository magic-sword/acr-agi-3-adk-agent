"""Host-only visual pointing. Movement never steps the environment or changes evidence."""
import base64
import io

from PIL import Image, ImageDraw

from agent.rendering import ORIGIN, SCALE, frame_image, render_current, png_base64


def start_cursor(observation, query):
    width, height = observation['width'], observation['height']
    stride = 1
    while stride * 2 <= max(width, height) // 8:
        stride *= 2
    return {'observation_id': observation['observation_id'], 'target_query': query,
            'mode':'locate', 'x':None, 'y':None,
            'stride': stride, 'adjustments': 0, 'confirmed': False}


def cursor_options(cursor, observation):
    width, height = observation['width'], observation['height']
    if cursor['mode']=='locate':
        # Region selection is a visual query, not an object inventory. Positions
        # become defined only after the model chooses a region of this frame.
        mx, my = (width+1)//2, (height+1)//2
        regions=[('1','upper left',(0,0,mx,my)), ('2','upper right',(mx,0,width,my)),
                 ('3','lower left',(0,my,mx,height)), ('4','lower right',(mx,my,width,height))]
        return {label:{'kind':'locate_cursor','region':name,'x':(left+right-1)//2,'y':(top+bottom-1)//2}
                for label,name,(left,top,right,bottom) in regions if left<right and top<bottom}
    x, y, s = cursor['x'], cursor['y'], cursor['stride']
    options = {}
    for label, direction, nx, ny in [('1','up',x,max(0,y-s)), ('2','down',x,min(height-1,y+s)),
                                    ('3','left',max(0,x-s),y), ('4','right',min(width-1,x+s),y)]:
        if (nx,ny) != (x,y):
            options[label] = {'kind':'move_cursor', 'direction':direction, 'x':nx, 'y':ny}
    if s > 1:
        options['5'] = {'kind':'resize_cursor', 'meaning':'smaller moves and closer view', 'stride':s//2}
    if s < max(width,height):
        options['6'] = {'kind':'resize_cursor', 'meaning':'larger moves and wider view', 'stride':s*2}
    options['7'] = {'kind':'click_cursor', 'meaning':'The requested instance and part are under the cursor'}
    return options


def adjust_cursor(cursor, option):
    if option['kind'] == 'locate_cursor':
        cursor.update(x=option['x'],y=option['y'],mode='adjust',region=option['region'])
        return
    if option['kind'] == 'move_cursor':
        cursor.update(x=option['x'], y=option['y'])
    elif option['kind'] == 'resize_cursor':
        cursor['stride'] = option['stride']
    else:
        raise ValueError('not a cursor adjustment')
    cursor['adjustments'] += 1


def observation_board(observation):
    if observation.get('grid') is not None:
        return frame_image(observation['grid'])
    if observation.get('_visual_frames'):
        return frame_image(observation['_visual_frames'][-1])
    # Only recover pixels from an explicitly mapped viewport, never guess how
    # an arbitrary screenshot maps to game coordinates.
    viewport = observation.get('viewport')
    if observation.get('image_png_base64') and viewport:
        image = Image.open(io.BytesIO(base64.b64decode(observation['image_png_base64']))).convert('RGB')
        ox, oy = viewport['origin']; scale = viewport['scale']
        width, height = observation['width'], observation['height']
        if scale <= 0 or ox < 0 or oy < 0 or ox+width*scale > image.width or oy+height*scale > image.height:
            raise ValueError('invalid cursor viewport')
        return image.crop((ox,oy,ox+width*scale,oy+height*scale)).resize((width,height),Image.Resampling.NEAREST)
    raise ValueError('cursor requires raw pixels or an explicitly mapped viewport')


def render_cursor(observation, cursor):
    if cursor['observation_id'] != observation['observation_id']:
        raise ValueError('stale cursor observation')
    board = observation_board(observation)
    x, y = cursor['x'], cursor['y']
    if type(x) is not int or type(y) is not int or not (0 <= x < board.width and 0 <= y < board.height):
        raise ValueError('cursor outside board')
    # Original pixels are immutable. Marks exist only on this independent preview.
    overview = render_current(board, label='HOST AIM: yellow marker is NOT a game object')
    ox, oy = ORIGIN
    draw = ImageDraw.Draw(overview)
    px, py = ox+x*SCALE, oy+y*SCALE
    draw.rectangle((px-2,py-2,px+SCALE+1,py+SCALE+1),outline='black',width=3)
    draw.rectangle((px-1,py-1,px+SCALE,py+SCALE),outline='#ffff00',width=1)
    radius = max(3, cursor['stride'] * 2)
    left, top = max(0,x-radius), max(0,y-radius)
    right, bottom = min(board.width,x+radius+1), min(board.height,y+radius+1)
    crop = board.crop((left,top,right,bottom))
    scale = max(1, min(24, 192//max(crop.size)))
    crop = crop.resize((crop.width*scale,crop.height*scale),Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(crop)
    cx, cy = (x-left)*scale, (y-top)*scale
    draw.rectangle((cx,cy,cx+scale-1,cy+scale-1),outline='#ffff00',width=min(2,scale))
    preview = Image.new('RGB',(overview.width+220,max(overview.height,270)),'#18202b')
    preview.paste(overview,(0,0));preview.paste(crop,(overview.width+12,76))
    draw = ImageDraw.Draw(preview)
    draw.text((overview.width+12,12),'HOST DETAIL: same frame',fill='white')
    draw.text((overview.width+12,30),f'cursor ({x},{y}), stride {cursor["stride"]}',fill='white')
    draw.text((overview.width+12,48),f'crop x={left}..{right-1}, y={top}..{bottom-1}',fill='white')
    return preview


def cursor_parts(observation, cursor):
    preview = render_cursor(observation,cursor)
    return [{'type':'text','text':'HOST AIM preview and detail; same frozen CURRENT board.'},
            {'type':'image_url','image_url':{'url':'data:image/png;base64,'+png_base64(preview)}}]
