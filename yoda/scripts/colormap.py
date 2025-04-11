

import numpy as np
hue_map_bright = {'CMCompare': '#023eff',
 'Infernal': '#ff7c00',
 'Infernal_global': '#1ac938',
 'KRAID': '#e8000b',
 'NSPDK': '#8b2be2',
 'random': '#9f4800'}

hue_map = {'CMCompare': '#4878d0', 'Infernal': '#ee854a', 'Infernal_global': '#6acc64', 'KRAID': '#d65f5f', 'NSPDK': '#956cb4', 'random': '#8c613c'}


hue_order =  [
 'KRAID',
  'CMCompare',
 'Infernal',
 'Infernal_global',
 'NSPDK',
 'random']


hue = {'palette':hue_map, 'hue_order':hue_order}

# made like this:
# z = sns.color_palette("muted")
# huemap = dict(zip(np.unique(df.Method),z.as_hex()))



def gethue(df,hue='Method'):
    ok =  np.unique(df[hue])
    hmap = dict(hue_map)
    horder = list(hue_order)
    for e in hue_map: 
        if e not in ok:
            hmap.pop(e)
            horder.remove(e)
            
    return {'palette':hmap, 'hue_order':horder}

    
        
        