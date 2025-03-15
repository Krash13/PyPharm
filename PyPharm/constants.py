MODEL_CONST = {
    'human':{
                  'adipose': {'V':143, 'Q': 60 * 3.7},
                  'bone': {'V':124, 'Q': 60 *  3.6} ,
                  'brain': {'V':20.7, 'Q': 60 *  10} ,
                  'gut': {'V':23.6, 'Q': 60 *  13} ,
                  'heart': {'V':3.8, 'Q': 60 *  2.14},
                  'kidney': {'V':4.4, 'Q': 60 *  15.7} ,
                  'liver': {'V':24.1, 'Q': 60 *  21} ,
                  'lung': {'V':16.7, 'Q': 60 *  71} ,
                  'muscle': {'V':429, 'Q': 60 *  10.7} ,
                  'pancreas': {'V':1.2, 'Q': 60 *  1.9} ,
                  'skin': {'V':111, 'Q': 60 *  4.3} ,
                  'spleen': {'V':2.7, 'Q': 60 *  1.1},
                  'stomach': {'V':2.2, 'Q': 60 * 0.56},
                  'teaster': {'V':0.51, 'Q': 60 * 0.04},
                  'arterial_blood': {'V':25.7} ,
                  'venous_blood': {'V':51.4}
               },
               'rat':{
                  'adipose': {'V':40, 'Q': 60 * 1.6},
                  'bone': {'V':53.2, 'Q': 60 *  10.12},
                  'brain': {'V':6.8, 'Q': 60 *  5.32}  ,
                  'gut': {'V':40, 'Q': 60 *  52}  ,
                  'heart': {'V':3.2, 'Q': 60 *  15.68},
                  'kidney': {'V':9.2, 'Q': 60 *  36.92},
                  'liver': {'V':41.2, 'Q': 60 *  80}  ,
                  'lung': {'V':4, 'Q': 60 *  203.2}   ,
                  'muscle': {'V':487.6, 'Q': 60 *  30} ,
                  'pancreas': {'V':5.2, 'Q': 60 *  4}  ,
                  'skin': {'V':160, 'Q': 60 *  20}   ,
                  'spleen': {'V':2.4, 'Q': 60 *  5}  ,
                  'stomach': {'V':4.4, 'Q': 60 * 8.2} ,
                  'teaster': {'V':10, 'Q': 60 * 1.8}   ,
                  'arterial_blood': {'V':22.4}  ,
                  # 'arterial_blood': {'V':22.4, 'Q': 60 * 10.8}  ,
                  'venous_blood': {'V':45.2}
               },
                'mouse': {
                    'adipose': {'V': 1000 * 1e-3 / 0.02, 'Q': 60 * 4e-5 * 1000 / 0.02},
                    'bone': {'V': 1000 * 1.58e-3 / 0.02, 'Q': 60 * 2.53e-4 * 1000 / 0.02},
                    'brain': {'V': 1000 * 1.7e-4 / 0.02, 'Q': 60 * 1.3e-4 * 1000 / 0.02},
                    'gut': {'V': 1000 * 6.27e-4 / 0.02, 'Q': 60 * 5e-4 * 1000 / 0.02},
                    'heart': {'V': 1000 * 9.5e-5 / 0.02, 'Q': 60 * 2.8e-4 * 1000 / 0.02},
                    'kidney': {'V': 1000 * 3.4e-4 / 0.02, 'Q': 60 * 1.3e-3 * 1000 / 0.02},
                    'liver': {'V': 1000 * 1e-3/ 0.02, 'Q' : 60 * 3.5e-4 * 1000 / 0.02},
                    'lung': {'V': 1000 * 1e-4 / 0.02, 'Q': 60 * 5.47e-3 * 1000 / 0.02},
                    'muscle': {'V': 1000 * 0.01e-1 / 0.02, 'Q': 60 * 9.1e-4 * 1000 / 0.02},
                    'pancreas': {'V': 1000 * 1.3e-4 / 0.02, 'Q': 60 * 5.2e-5 * 1000 / 0.02},
                    'skin': {'V': 1000 * 2.9e-3 / 0.02, 'Q': 60 * 4.1e-4 * 1000 / 0.02},
                    'spleen': {'V': 1000 * 1e-4 / 0.02, 'Q': 60 * 9e-5 * 1000 / 0.02},
                    'stomach': {'V': 1000 * 1.1e-4 / 0.02, 'Q': 60 * 1.1e-4 * 1000 / 0.02},
                    'teaster': {'V': 1, 'Q': 1},
                    'arterial_blood': {'V': 1000 * 2.28e-4 / 0.02},
                    # 'arterial_blood': {'V':22.4, 'Q': 60 * 10.8}  ,
                    'venous_blood': {'V': 1000 * 5.25e-4 / 0.02}
                }
}

class ORGAN_NAMES:

    LUNG = 'lung'
    HEART = 'heart'
    BRAIN = 'brain'
    MUSCLE = 'muscle'
    ADIPOSE = 'adipose'
    SKIN = 'skin'
    BONE = 'bone'
    KIDNEY = 'kidney'
    LIVER = 'liver'
    GUT = 'gut'
    SPLEEN = 'spleen'
    STOMACH = 'stomach'
    PANCREAS = 'pancreas'
    VENOUS = 'venous_blood'
    ARTERIAL = 'arterial_blood'

class ANIMALS:
    HUMAN = 'human'
    RAT = 'rat'
    MOUSE = 'mouse'