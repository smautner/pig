
hue_map = {'CMCompare': '#023eff',
 'Infernal': '#ff7c00',
 'Infernal_global': '#1ac938',
 'KRAID': '#e8000b',
 'NSPDK': '#8b2be2',
 'random': '#9f4800'}

hue_order =  [
 'KRAID',
  'CMCompare',
 'Infernal',
 'Infernal_global',
 'NSPDK',
 'random']


hue = {'palette':hue_map, 'hue_order':hue_order}
# made like this:
# z = sns.color_palette("bright")
# huemap = dict(zip(np.unique(df.Method),z.as_hex()))