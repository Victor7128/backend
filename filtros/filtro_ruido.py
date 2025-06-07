import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from fastapi import APIRouter, File, UploadFile, HTTPException

router = APIRouter()

# CONSTANTES DE CARACTERÍSTICAS DE PLANTILLAS PRECALCULADAS
PLANTILLAS_CARACTERISTICAS = [
    {
        'gradient_stats': {
            'mean': 6.37910077518333,
            'std': 36.99868554672559,
            'max': 629.4966242959529,
            'area_percentage': 15.783114558472553
        },
        'edge_density': 0.019487432875894987,
        'shape_stats': {
            'solidity': 0,
            'rectangularity': 1,
            'area_ratio': 0
        },
        'texture_histogram': [0.005111687052505967, 0.00030578758949880667, 9.136336515513127e-05, 9.322792362768496e-05, 0.00046147822195704056, 4.3817124105011936e-05, 0.00014916467780429595, 0.0006647150954653938, 9.322792362768496e-05, 6.0598150357995224e-05, 0.0003440110381861575, 2.1442422434367542e-05, 0.00014916467780429595, 4.474940334128878e-05, 0.0008502386634844869, 0.0019009173627684965, 0.000278751491646778, 0.001705138723150358, 6.898866348448687e-05, 5.034307875894988e-05, 5.313991646778043e-05, 7.458233890214797e-06, 5.034307875894988e-05, 7.365005966587112e-05, 0.00011467034606205251, 4.008800715990453e-05, 2.237470167064439e-05, 3.542661097852029e-05, 0.0006796315632458233, 6.619182577565633e-05, 0.001965244630071599, 0.017779497315035798, 9.322792362768496e-05, 5.500447494033413e-05, 3.729116945107399e-05, 2.703609785202864e-05, 9.043108591885442e-05, 1.3984188544152745e-05, 3.4494331742243436e-05, 0.00010068615751789976, 5.593675417661098e-06, 1.5848747016706442e-05, 1.8645584725536993e-06, 7.458233890214797e-06, 2.703609785202864e-05, 8.390513126491648e-06, 2.890065632458234e-05, 0.00011746718377088305, 0.0001482323985680191, 6.246270883054893e-05, 2.796837708830549e-05, 2.610381861575179e-05, 6.898866348448687e-05, 4.6613961813842484e-06, 6.898866348448687e-05, 4.847852028639618e-05, 0.000514618138424821, 3.4494331742243436e-05, 3.729116945107399e-05, 4.195256563245823e-05, 0.0027427655131264917, 3.822344868735083e-05, 0.0011299224343675417, 0.0009434665871121718, 0.00046147822195704056, 5.313991646778043e-05, 8.856652744630071e-05, 8.390513126491646e-05, 0.003113812649164678, 1.0255071599045346e-05, 8.390513126491646e-05, 0.00011560262529832936, 9.136336515513127e-05, 7.458233890214797e-06, 2.983293556085919e-05, 2.1442422434367542e-05, 8.483741050119331e-05, 8.390513126491648e-06, 6.0598150357995224e-05, 5.593675417661098e-05, 4.941079952267303e-05, 7.458233890214797e-06, 1.7713305489260144e-05, 5.593675417661098e-06, 9.322792362768497e-06, 9.322792362768497e-07, 9.322792362768497e-06, 8.390513126491648e-06, 8.576968973747017e-05, 7.458233890214797e-06, 1.3984188544152745e-05, 4.1020286396181385e-05, 0.00011467034606205251, 9.322792362768497e-06, 5.407219570405728e-05, 0.0005286023269689738, 0.00015475835322195704, 5.8733591885441526e-05, 4.5681682577565635e-05, 6.0598150357995224e-05, 8.017601431980906e-05, 9.322792362768497e-06, 3.356205250596659e-05, 7.644689737470167e-05, 2.610381861575179e-05, 9.322792362768497e-06, 4.6613961813842484e-06, 1.3984188544152745e-05, 5.686903341288783e-05, 1.3984188544152745e-05, 4.5681682577565635e-05, 0.00015289379474940333, 0.000657256861575179, 7.085322195704058e-05, 6.712410501193318e-05, 6.898866348448687e-05, 0.00011187350835322196, 1.4916467780429595e-05, 7.924373508353222e-05, 0.00014170644391408115, 0.0025581742243436754, 3.822344868735083e-05, 0.0001743362171837709, 0.00017713305489260142, 0.007362209128878282, 0.00012026402147971361, 0.0009238887231503579, 0.0017340393794749403, 7.551461813842483e-05, 0.0001221285799522673, 1.0255071599045346e-05, 0.0005341960023866349, 8.670196897374702e-05, 6.898866348448687e-05, 2.330698090692124e-05, 0.002397822195704057, 3.076521479713604e-05, 2.890065632458234e-05, 9.322792362768497e-07, 3.262977326968974e-05, 3.6358890214797135e-05, 7.365005966587112e-05, 1.8645584725536994e-05, 0.0011178028042959428, 7.085322195704058e-05, 5.034307875894988e-05, 1.2119630071599045e-05, 3.356205250596659e-05, 1.0255071599045346e-05, 6.525954653937947e-06, 8.390513126491648e-06, 4.3817124105011936e-05, 3.542661097852029e-05, 2.703609785202864e-05, 3.7291169451073987e-06, 4.7546241050119334e-05, 7.644689737470167e-05, 8.110829355608592e-05, 0.0001090766706443914, 0.0009378729116945108, 0.00034214647971360384, 2.330698090692124e-05, 0.0, 2.983293556085919e-05, 2.0510143198090692e-05, 1.5848747016706442e-05, 4.6613961813842484e-06, 0.00016035202863961813, 5.593675417661098e-06, 1.1187350835322196e-05, 3.7291169451073987e-06, 7.458233890214797e-06, 3.7291169451073987e-06, 1.0255071599045346e-05, 1.8645584725536994e-05, 9.043108591885442e-05, 2.1442422434367542e-05, 5.034307875894988e-05, 4.6613961813842484e-06, 3.6358890214797135e-05, 1.9577863961813843e-05, 3.262977326968974e-05, 1.2119630071599045e-05, 0.00017806533412887828, 2.0510143198090692e-05, 5.034307875894988e-05, 1.1187350835322196e-05, 0.00018272673031026254, 0.00017060710023866348, 0.00017899761336515513, 8.483741050119331e-05, 0.0017098001193317422, 0.000142638723150358, 0.0006022523866348449, 2.330698090692124e-05, 0.0029049821002386637, 8.017601431980906e-05, 0.00010814439140811456, 4.847852028639618e-05, 0.007412552207637231, 3.076521479713604e-05, 7.737917661097852e-05, 3.7291169451073987e-06, 0.00019484636038186157, 4.195256563245823e-05, 7.365005966587112e-05, 3.1697494033412885e-05, 0.0009416020286396182, 5.686903341288783e-05, 6.898866348448687e-05, 1.4916467780429595e-05, 5.127535799522673e-05, 3.7291169451073987e-06, 1.3051909307875894e-05, 1.6781026252983295e-05, 9.788931980906921e-05, 5.686903341288783e-05, 8.297285202863962e-05, 1.2119630071599045e-05, 0.00016408114558472554, 8.017601431980906e-05, 0.00016221658711217183, 0.00013424821002386636, 0.0017890438544152746, 0.0008362544749403342, 0.0020286396181384246, 1.6781026252983295e-05, 0.0011243287589498808, 5.966587112171838e-05, 5.686903341288783e-05, 3.542661097852029e-05, 0.0009164304892601432, 2.703609785202864e-05, 0.0001482323985680191, 4.6613961813842484e-06, 9.322792362768496e-05, 5.407219570405728e-05, 0.00014170644391408115, 0.0001482323985680191, 0.0010609337708830549, 0.0018738812649164679, 0.010798590393794749, 0.0001295868138424821, 0.0010003356205250597, 7.365005966587112e-05, 0.0005276700477326969, 0.00014450328162291169, 0.0017573463603818616, 0.0011839946300715991, 0.0009863514319809069, 9.882159904534606e-05, 0.001610978520286396, 0.0009313469570405728, 0.0017759919451073986, 0.0009705026849642005, 0.8836795196897375],
        'spectral_stats': {
            'mean_magnitude': 10872.163879984144,
            'std_magnitude': 193599.10222303917,
            'energy': 4.032999434713857e+16
        },
        'image_size': (1280, 838)
    },
    {
        'gradient_stats': {
            'mean': 8.355971067784973,
            'std': 42.32036741968164,
            'max': 754.0026525152282,
            'area_percentage': 11.259058873544436
        },
        'edge_density': 0.015219903947709293,
        'shape_stats': {
            'solidity': 0,
            'rectangularity': 1,
            'area_ratio': 0
        },
        'texture_histogram': [0.004680058189354536, 0.0009301329188338216, 0.00032183495745401647, 0.00032233315398258304, 0.0010257866523186066, 0.0001489607620414101, 0.00033678085331101414, 0.000837468364520436, 0.00033130069149678163, 0.00022369024132639845, 6.924931747075586e-05, 6.277276259939021e-05, 0.0003352862637253144, 0.00017685976764113907, 0.001098025148960762, 0.002051573304637213, 0.0009301329188338216, 0.005913592794085411, 0.0002341523684262968, 0.00016340846136984118, 0.00017486698152687272, 5.181243897092525e-05, 0.00017636157111257248, 0.00014348060022717761, 0.0002979215240828202, 0.00018582730515533767, 6.028177995655726e-05, 7.72204619278213e-05, 0.0009371076702337538, 0.00014696797592714375, 0.0021870827604073256, 0.017350192303860028, 0.0003238277435682828, 0.00023813794065482953, 7.522767581355493e-05, 9.166816125625236e-05, 0.00023116318925489729, 4.4837687570993004e-05, 8.020964109922082e-05, 0.00016988501624120683, 6.127817301369044e-05, 8.170423068492059e-05, 1.6938681971264023e-05, 2.5408022956896036e-05, 7.273669317072198e-05, 6.127817301369044e-05, 6.028177995655726e-05, 0.0002251848309120982, 0.0003113728303541181, 0.00017038321276977342, 9.41591438990853e-05, 8.867898208485283e-05, 0.00015892469261274188, 2.5408022956896036e-05, 0.00012853470437017994, 0.00014646977939857715, 0.0006237420537653694, 0.00011010143281321616, 6.277276259939021e-05, 7.72204619278213e-05, 0.0026509037285028197, 8.967537514198601e-05, 0.0009839381439190132, 0.0012151013331739105, 0.0010516928718040692, 0.0001524481377413762, 0.0002301667961977641, 0.00017337239194117296, 0.004545046930112991, 4.0852115342460295e-05, 0.00017038321276977342, 0.00018383451904107131, 0.0002256830274406648, 4.583408062812618e-05, 6.426735218508997e-05, 4.8823259799525714e-05, 0.00016839042665550707, 2.5906219485462624e-05, 0.00010860684322751639, 0.00010711225364181663, 0.00015045535162710986, 4.433949104242642e-05, 4.6332277156692775e-05, 2.939359518542875e-05, 4.8325063270959126e-05, 6.974751399932245e-06, 3.3877363942528046e-05, 1.5942288914130847e-05, 0.0001853291086267711, 2.5906219485462624e-05, 5.330702855662502e-05, 5.9285386899424085e-05, 0.00016739403359837388, 1.6440485442697435e-05, 9.266455431338555e-05, 0.0005226081584663518, 0.0003522249456965784, 0.00016888862318407364, 9.067176819911918e-05, 0.00013899683147007832, 0.00017436878499830613, 2.6902612542595803e-05, 9.116996472768578e-05, 0.00012504732867021383, 6.376915565652339e-05, 5.081604591379207e-05, 2.5408022956896036e-05, 3.437556047109464e-05, 7.72204619278213e-05, 6.177636954225702e-05, 9.46573404276519e-05, 0.0002665351427831251, 0.0007712082262210797, 0.00015493912038420915, 0.00017287419541260637, 0.00013700404535581197, 0.00017088140929834002, 1.893146808553038e-05, 0.0001479643689842769, 0.00011857077379884817, 0.0024237261114764554, 9.41591438990853e-05, 0.00019678762878380264, 0.00022418843785496502, 0.008470337378689146, 0.0001883182877981706, 0.0011099818656463603, 0.0015703154580418884, 0.0003382754428967139, 0.000313863812996951, 6.725653135648951e-05, 0.0006097925509655048, 0.00023763974412626293, 0.00017387058846973953, 6.924931747075586e-05, 0.002338036308563002, 7.821685498495447e-05, 8.668619597058648e-05, 1.6938681971264023e-05, 6.476554871365656e-05, 9.714832307048485e-05, 0.00012205814949881429, 6.376915565652339e-05, 0.0009206671847910564, 0.0002306649927263307, 0.00016739403359837388, 8.1206034156354e-05, 0.00010611586058468344, 4.8823259799525714e-05, 2.740080907116239e-05, 6.825292441362269e-05, 9.216635778481895e-05, 8.768258902771966e-05, 7.67222653992547e-05, 2.391343337119627e-05, 8.668619597058648e-05, 0.00017636157111257248, 0.00013351666965584583, 0.0002296685996691975, 0.0012076283852454116, 7.323488969928858e-05, 6.127817301369044e-05, 1.6440485442697435e-05, 6.277276259939021e-05, 6.875112094218928e-05, 5.779079731372432e-05, 2.0426057671230148e-05, 0.00020226779059803512, 1.4945895856997668e-05, 2.0426057671230148e-05, 5.480161814232479e-06, 2.0924254199796735e-05, 2.2418843785496502e-05, 4.035391881389371e-05, 2.4411629899762857e-05, 0.00014646977939857715, 5.231063549949184e-05, 7.024571052788904e-05, 1.9927861142663556e-05, 8.419521332775354e-05, 5.280883202805843e-05, 5.330702855662502e-05, 3.985572228532711e-05, 0.00021920647256929912, 6.426735218508997e-05, 8.668619597058648e-05, 1.9927861142663556e-05, 0.0001933002530838365, 0.0002117335246408003, 0.00024511269205476177, 0.00014547338634144397, 0.0019409736752954306, 0.0003298061019110819, 0.0008270062374205377, 6.476554871365656e-05, 0.002692254040373847, 0.00017287419541260637, 0.00017337239194117296, 7.971144457065423e-05, 0.008240170582491381, 9.067176819911918e-05, 0.00016839042665550707, 1.6440485442697435e-05, 0.00019927861142663558, 8.270062374205377e-05, 0.0001130906119846157, 7.572587234212152e-05, 0.0010910503975608298, 0.0001893146808553038, 0.00014696797592714375, 6.376915565652339e-05, 8.818078555628624e-05, 2.291704031406309e-05, 1.7935075028397202e-05, 6.626013829935633e-05, 0.00019678762878380264, 0.0001439787967557442, 0.00013401486618441243, 3.786293617106076e-05, 0.00024511269205476177, 0.00011956716685598135, 0.00012753831131304678, 0.00025009465734042766, 0.0015867559434845858, 0.0008673601562344314, 0.0023205994300631713, 6.177636954225702e-05, 0.0009679958550048824, 0.00011607979115601522, 8.917717861341943e-05, 8.369701679918694e-05, 0.0009729778202905482, 6.875112094218928e-05, 0.00022867220661206434, 2.1920647256929914e-05, 0.0001534445307985094, 8.867898208485283e-05, 0.00027101891154022436, 0.0002256830274406648, 0.0017436878499830614, 0.0020431039636515814, 0.014898068990255276, 0.00021522090034076642, 0.0011059962934178275, 9.714832307048485e-05, 0.00031336561646838444, 0.00025906219485462626, 0.0014721707419142703, 0.0009256491500767223, 0.0011079890795320938, 0.0001544409238556426, 0.0017132978617404994, 0.0009859309300332794, 0.0015000697475139994, 0.0017611247284828918, 0.8550776190191507],
        'spectral_stats': {
            'mean_magnitude': 13273.915404046167,
            'std_magnitude': 293615.2007642762,
            'energy': 1.7339760114055798e+17
        },
        'image_size': (1556, 1290)
    }
]

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

def extraer_caracteristicas_imagen(image: np.ndarray) -> Dict[str, Any]:
    """Extrae características de una imagen para comparación"""
    
    # 1. Análisis de color y gradiente en región morada
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([125, 50, 50])
    upper_purple = np.array([135, 255, 255])  # Rango más específico
    mask_purple = cv2.inRange(hsv, lower_purple, upper_purple)
    
    purple_region = cv2.bitwise_and(image, image, mask=mask_purple)
    purple_gray = cv2.cvtColor(purple_region, cv2.COLOR_BGR2GRAY)
    
    # Gradientes
    grad_x = cv2.Sobel(purple_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(purple_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    purple_pixels = purple_gray > 0
    if np.sum(purple_pixels) > 0:
        gradient_stats = {
            'mean': float(np.mean(gradient_magnitude[purple_pixels])),
            'std': float(np.std(gradient_magnitude[purple_pixels])),
            'max': float(np.max(gradient_magnitude[purple_pixels])),
            'area_percentage': float(np.sum(purple_pixels) / image.size * 100)
        }
    else:
        gradient_stats = {'mean': 0, 'std': 0, 'max': 0, 'area_percentage': 0}
    
    # 2. Análisis de estructura y bordes
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Densidad de bordes
    edge_density = float(np.sum(edges > 0) / edges.size)
    
    # 3. Análisis de forma principal (tarjeta blanca)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shape_stats = {'solidity': 0.0, 'rectangularity': 0.0, 'area_ratio': 0.0}
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area > 1000:  # Solo considerar contornos significativos
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            shape_stats['solidity'] = float(area / hull_area) if hull_area > 0 else 0.0
            
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            shape_stats['rectangularity'] = 1.0 if len(approx) == 4 else 0.0
            shape_stats['area_ratio'] = float(area / image.size)
    
    # 4. Análisis de textura usando LBP simplificado
    def calcular_lbp_histogram_rapido(img_gray):
        # Redimensionar para acelerar el cálculo
        h, w = img_gray.shape
        if h > 400 or w > 400:
            scale = min(400/h, 400/w)
            new_h, new_w = int(h*scale), int(w*scale)
            img_gray = cv2.resize(img_gray, (new_w, new_h))
        
        height, width = img_gray.shape
        lbp = np.zeros((height-2, width-2), dtype=np.uint8)
        
        # Vectorizar el cálculo LBP
        for i in range(1, height-1):
            center = img_gray[i, 1:width-1]
            pattern = np.zeros_like(center, dtype=np.uint8)
            
            # Calcular patrón binario local
            pattern += (img_gray[i-1, 0:width-2] >= center).astype(np.uint8) << 7
            pattern += (img_gray[i-1, 1:width-1] >= center).astype(np.uint8) << 6
            pattern += (img_gray[i-1, 2:width] >= center).astype(np.uint8) << 5
            pattern += (img_gray[i, 2:width] >= center).astype(np.uint8) << 4
            pattern += (img_gray[i+1, 2:width] >= center).astype(np.uint8) << 3
            pattern += (img_gray[i+1, 1:width-1] >= center).astype(np.uint8) << 2
            pattern += (img_gray[i+1, 0:width-2] >= center).astype(np.uint8) << 1
            pattern += (img_gray[i, 0:width-2] >= center).astype(np.uint8) << 0
            
            lbp[i-1, :] = pattern
        
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        return hist / np.sum(hist)  # Normalizar
    
    texture_histogram = calcular_lbp_histogram_rapido(gray)
    
    # 5. Análisis espectral simplificado
    # Usar una región central más pequeña para acelerar FFT
    h, w = gray.shape
    center_region = gray[h//4:3*h//4, w//4:3*w//4]
    
    f_transform = np.fft.fft2(center_region)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Características espectrales
    spectral_stats = {
        'mean_magnitude': float(np.mean(magnitude_spectrum)),
        'std_magnitude': float(np.std(magnitude_spectrum)),
        'energy': float(np.sum(magnitude_spectrum**2))
    }
    
    return {
        'gradient_stats': gradient_stats,
        'edge_density': edge_density,
        'shape_stats': shape_stats,
        'texture_histogram': texture_histogram.tolist(),
        'spectral_stats': spectral_stats,
        'image_size': image.shape[:2]
    }

def comparar_con_plantillas(caracteristicas_test: Dict) -> Dict[str, Any]:
    """Compara las características de la imagen test con las plantillas precalculadas"""
    
    similitudes = []
    
    for i, template_features in enumerate(PLANTILLAS_CARACTERISTICAS):
        scores = {}
        
        # 1. Similitud de gradiente (peso: 0.30)
        grad_test = caracteristicas_test['gradient_stats']
        grad_template = template_features['gradient_stats']
        
        if grad_template['mean'] > 0:
            grad_mean_sim = 1.0 - min(1.0, abs(grad_test['mean'] - grad_template['mean']) / max(grad_template['mean'], 10))
            grad_std_sim = 1.0 - min(1.0, abs(grad_test['std'] - grad_template['std']) / max(grad_template['std'], 10))
            grad_area_sim = 1.0 - min(1.0, abs(grad_test['area_percentage'] - grad_template['area_percentage']) / 30.0)
            
            gradient_similarity = (grad_mean_sim + grad_std_sim + grad_area_sim) / 3
        else:
            gradient_similarity = 0.5 if grad_test['area_percentage'] == 0 else 0.2
        
        scores['gradient'] = gradient_similarity
        
        # 2. Similitud de estructura (peso: 0.25)
        shape_test = caracteristicas_test['shape_stats']
        shape_template = template_features['shape_stats']
        
        # Para imágenes Yape, la rectangularidad debería ser consistente
        rect_similarity = 1.0 if shape_test['rectangularity'] == shape_template['rectangularity'] else 0.3
        
        # Similitud de solidez (si ambas tienen contornos válidos)
        if shape_test['area_ratio'] > 0 and shape_template['area_ratio'] > 0:
            solidity_sim = 1.0 - min(1.0, abs(shape_test['solidity'] - shape_template['solidity']))
        else:
            solidity_sim = 0.5
        
        structure_similarity = (rect_similarity + solidity_sim) / 2
        scores['structure'] = structure_similarity
        
        # 3. Similitud de textura usando correlación de histogramas (peso: 0.25)
        hist_test = np.array(caracteristicas_test['texture_histogram'])
        hist_template = np.array(template_features['texture_histogram'])
        
        # Usar correlación de Pearson para comparar histogramas
        correlation = np.corrcoef(hist_test, hist_template)[0, 1]
        texture_similarity = max(0, correlation) if not np.isnan(correlation) else 0
        scores['texture'] = texture_similarity
        
        # 4. Similitud de densidad de bordes (peso: 0.20)
        edge_test = caracteristicas_test['edge_density']
        edge_template = template_features['edge_density']
        
        edge_similarity = 1.0 - min(1.0, abs(edge_test - edge_template) / max(edge_template, 0.01))
        scores['edges'] = edge_similarity
        
        # Calcular similitud ponderada total
        pesos = {
            'gradient': 0.30,
            'structure': 0.25,
            'texture': 0.25,
            'edges': 0.20
        }
        
        similitud_total = sum(scores[key] * pesos[key] for key in pesos.keys())
        similitudes.append({
            'template_index': i,
            'similarity_score': similitud_total,
            'detailed_scores': scores
        })
    
    # Encontrar la mejor coincidencia
    mejor_match = max(similitudes, key=lambda x: x['similarity_score'])
    
    return {
        'best_match_similarity': mejor_match['similarity_score'],
        'best_template_index': mejor_match['template_index'],
        'all_similarities': similitudes,
        'average_similarity': np.mean([s['similarity_score'] for s in similitudes])
    }

def calcular_porcentaje_veracidad_optimizado(similitud_info: Dict) -> float:
    """Calcula el porcentaje de veracidad basado en similitud con plantillas"""
    
    best_similarity = similitud_info['best_match_similarity']
    avg_similarity = similitud_info['average_similarity']
    
    # Dar más peso a la mejor similitud
    combined_similarity = (best_similarity * 0.8) + (avg_similarity * 0.2)
    
    # Mapeo más agresivo para distinguir mejor entre auténtico y falso
    if combined_similarity >= 0.75:
        porcentaje_veracidad = 80 + (combined_similarity - 0.75) * 60  # 80-95%
    elif combined_similarity >= 0.60:
        porcentaje_veracidad = 60 + (combined_similarity - 0.60) * 133.33  # 60-80%
    elif combined_similarity >= 0.40:
        porcentaje_veracidad = 30 + (combined_similarity - 0.40) * 150  # 30-60%
    elif combined_similarity >= 0.25:
        porcentaje_veracidad = 15 + (combined_similarity - 0.25) * 100  # 15-30%
    else:
        porcentaje_veracidad = combined_similarity * 60  # 0-15%
    
    return max(5.0, min(95.0, porcentaje_veracidad))

# Endpoint opcional para obtener información detallada
@router.post("/filtro_ruido_detallado")
async def filtro_ruido_detallado(file: UploadFile = File(...)):
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=422, detail="❌ El archivo debe ser una imagen")
        
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=422, detail="❌ No se pudo decodificar la imagen")
        
        caracteristicas_test = extraer_caracteristicas_imagen(image)
        similitud_info = comparar_con_plantillas(caracteristicas_test)
        porcentaje_veracidad = calcular_porcentaje_veracidad_optimizado(similitud_info)
        
        return {
            "porcentaje_veracidad": f"{porcentaje_veracidad:.2f}%",
            "mejor_similitud": f"{similitud_info['best_match_similarity']*100:.2f}%",
            "plantilla_coincidente": f"Plantilla {similitud_info['best_template_index'] + 1}",
            "similitud_promedio": f"{similitud_info['average_similarity']*100:.2f}%",
            "caracteristicas_detectadas": {
                "area_morada": f"{caracteristicas_test['gradient_stats']['area_percentage']:.2f}%",
                "densidad_bordes": f"{caracteristicas_test['edge_density']:.4f}",
                "forma_rectangular": caracteristicas_test['shape_stats']['rectangularity'] == 1.0
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analizando imagen Yape: {e}")