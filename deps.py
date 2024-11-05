import warnings
from typing import Literal as Lit
import datetime as dt
import itertools as it
from numpy import nan, exp, log
import numpy as np
import pandas as pd
import yfinance as yf



# ============================================ #
# ================ DTYPE CAST ================ #
# ============================================ #
def py_int(x, method:Lit['int','round']='round', 
    err_rtn:Lit['entry','na','alt']='na', alt=..., catch=True, logErr=False
):
    try:
        if method == 'int':     return   int(x)
        if method == 'round':   return round(x)

    except Exception as Error:
        if not catch: raise Error
        if logErr: print('Value:',x, 'Error:',Error)
        
        if err_rtn == 'entry':  return x
        if err_rtn == 'na':     return pd.NA
        if err_rtn == 'alt':    return alt

def np_int(X, method:Lit['int','round']='round', 
    alt=pd.NA, err_val:Lit['entry','alt']='alt', catch=True, logErr=False, 
    to:Lit['list','numpy','pandas']='list'
):
    Y = []
    for x in X:
        try:
            if method == 'int':     Y.append(int(x))
            if method == 'round':   Y.append(round(x))
            
        except Exception as Error:
            if not catch: raise Error
            if logErr: print(Error)
            if err_val == 'entry':  Y.append(x)
            if err_val == 'alt':    Y.append(alt)

    if to == 'pandas':  return pd.Series(Y)
    if to == 'numpy':   return np.array(Y)
    else:               return Y

def _Cast(Data, method:Lit['dot_astype','to_numeric','map_numeric'], 
    dtype='float', errors:Lit['ignore','coerce','raise']='ignore', 
    skip=False, logErr=True
):
    if skip: return Data

    try:
        if method == 'dot_astype':  return Data.astype(dtype)
        if method == 'to_numeric':  return pd.to_numeric(Data, errors=errors)
        if method == 'map_numeric': return Data.apply(pd.to_numeric, errors=errors)

    except Exception as error:
        if logErr: print(error)
        return Data

def strip_pct(Data, cast:Lit['skip','dot_astype','to_numeric','map_numeric'], 
    char='%', dtype='float', errors:Lit['ignore','coerce','raise']='coerce', logErr=False
):
    try:
        Striped = Data.str.rstrip(char)

        if (cast == 'skip'):    return Striped
        if (cast != 'skip'):    return _Cast(Striped, method=cast, dtype=dtype, errors=errors)

    except Exception as error:
        if logErr: print(error)
        return np.nan


# ======================================== #
# ================ SERIES ================ #
# ======================================== #
def Finally(Data, mult=None, fig=None, logErr=True):
    try:    
        pipe = None
    
        if (mult and fig):  pipe = (Data * mult).round(fig)
        elif (mult):        pipe = (Data * mult)
        elif (fig):         pipe = (Data       ).round(fig)
        else:               pipe = (Data       )

        return pipe

    except Exception as error:
        if logErr: print(error)
        return Data


# =========================================== #
# ================ SELECTORS ================ #
# =========================================== #
def pd_getter(Df, Col, 
    _as:Lit['same','date','i64','num','abs','pct','str']='same', errors='coerce', 
    mult=None, fig=None, err_rtn:Lit['void','none','nan','na','nat','','alt']='nan', alt=..., catch=True, logErr=False
):
    try:    
        pipe = None
    
        if (_as == 'same'):     pipe = Df[Col]
        if (_as == 'date'):     pipe = Df[Col].astype('datetime64[ns]')
        if (_as == 'i64' ):     pipe = np_int(Df[Col])
        if (_as == 'num' ):     pipe = _Cast(Df[Col],    method='to_numeric', errors=errors)
        if (_as == 'abs' ):     pipe = _Cast(Df[Col],    method='to_numeric', errors=errors).abs()
        if (_as == 'pct' ):     pipe = strip_pct(Df[Col], cast='to_numeric')
        if (_as == 'str' ):     pipe = Df[Col].astype(str)

        return Finally(pipe, mult, fig)

    except Exception as Error:
        if not catch: raise Error
        if logErr: print('Col:',Col, 'Error:',Error)
    
        if err_rtn == 'none':   return None
        if err_rtn == 'nan':    return np.nan
        if err_rtn == 'na':     return pd.NA
        if err_rtn == 'nat':    return pd.NaT
        if err_rtn == '':       return ""
        if err_rtn == 'alt':    return alt


# ========================================== #
# ================ COALESCE ================ #
# ========================================== #
def pd_num_coalesce(Df, Labels, check:Lit['is_fin','is_geo']='is_fin',
    err_rtn:Lit['void','none','zero','nan','na','','alt']='nan', alt=..., logErr=True, 
    
    any_key_action: Lit['raise','err_rtn','try_next'] = 'try_next', 
    last_key_action:Lit['raise','err_rtn']            = 'raise', 

    any_val_action: Lit['raise','err_rtn','try_next'] = 'try_next', 
    last_val_action:Lit['raise','err_rtn']            = 'err_rtn', 

    ERR_KEY='COLUMN_DOES_NOT_EXIST', ERR_ALL_KEYS='ALL_COLUMNS_DOEST_NOT_EXIST', 
    ERR_VAL='MISSING_VALUE',         ERR_ALL_VALS='MISSING_ALL_VALUES'
):

    # ======================== Init Vars ======================== #
    Labels = np.ravel(Labels)

    # ======================== Helpers ======================== #
    def is_fin(x):  return isinstance(x, int|float) and (float('-inf') < x < float('+inf'))
    def is_geo(x):  return isinstance(x, int|float) and (            0 < x < float('inf'))

    def validator(x):
        if check == 'is_fin':   return is_fin(x)
        if check == 'is_geo':   return is_geo(x)

    def err_return(): 
        if err_rtn == 'none':   return None
        if err_rtn == 'zero':   return 0
        if err_rtn == 'nan':    return float('nan')
        if err_rtn == 'na':     return pd.NA
        if err_rtn == '':       return ''
        if err_rtn == 'alt':    return alt

    # ======================== Main ======================== #
    def Lambda(cols):
        N = len(Labels)
        for i, lab in enumerate(Labels):
            is_last   = (N == i+1)
            has_label = (lab in cols)

            if (not has_label) and (not is_last):
                if logErr:                          print(ERR_KEY)
                if any_key_action == 'raise':       raise Exception(ERR_KEY)
                if any_key_action == 'err_rtn':     return err_return()
                if any_key_action == 'try_next':    continue

            if (not has_label) and (is_last):
                if logErr:                          print(ERR_ALL_KEYS)
                if last_key_action == 'raise':      raise Exception(ERR_ALL_KEYS)
                if last_key_action == 'err_rtn':    return err_return()

            is_valid = validator(cols[lab])

            if (not is_valid) and (not is_last):
                if logErr:                          print(ERR_VAL)
                if any_val_action == 'raise':       raise Exception(ERR_VAL)
                if any_val_action == 'err_rtn':     return err_return()
                if any_val_action == 'try_next':    continue

            if (not is_valid) and (is_last):
                if logErr:                          print(ERR_ALL_VALS)
                if last_val_action == 'raise':      raise Exception(ERR_ALL_VALS)
                if last_val_action == 'err_rtn':    return err_return()

            if is_valid:                            
                return cols[lab]

    # ======================== Application ======================== #
    if isinstance(Df, pd.DataFrame):    return Df.apply(Lambda, axis=1)
    else:                               return Lambda


# ============================================ #
# ================ DATAFRAMES ================ #
# ============================================ #
class pd_DataFrame_fromSecs():
    def __init__(my, Sections):
        my.Sections = Sections

    def append(my, Sec):
        my.Sections.append(Sec)

    def concat(my, axis=0, *args, **kwargs):
        return pd.concat(my.Sections, axis=axis, *args, **kwargs)
    
    def concat_reset(my, axis=0, drop=1, *args, **kwargs):
        return my.concat(axis=axis).reset_index(drop=drop, *args, **kwargs)

def pd_reduce_Secs(Keys, Lambda, Initial, axis=0, drop=1, verbose=True):
    
    Df = pd_DataFrame_fromSecs(Initial)
    L  = len(Keys)

    for i, key in enumerate(Keys):
        if verbose:  print('Start,', f'iLoop: {i}/{L},', 'Key:',key)
        Df.append(Lambda(i, key))
        if verbose:  print('Finish,', f'iLoop: {i}/{L},', 'Key:',key)
    pass
    return Df.concat_reset(axis, drop)













# ========================================================================================= #
# ======================================== FINANCE ======================================== #
# ========================================================================================= #

TECH_1  = ['MSFT']
TECH_3  = ['MSFT','AAPL','GOOG']
TECH_5  = ['MSFT','AAPL','GOOG','META','AMZN']
TECH_7  = ['MSFT','AAPL','GOOG','AMZN','META','NVDA','NFLX']
BEST_20 = ['AAPL','MSFT','GOOGL','META','UNH','V','MA','INTU','CPRT','IDXX','ODFL','SHW','EW','ANET','RMD','WST','MPWR','POOL','EPAM','MKTX']
DJI_30  = ['UNH','MSFT','GS','HD','CAT','CRM','MCD','AMGN','V','TRV','AXP','BA','HON','IBM','JPM','AAPL','AMZN','JNJ','PG','CVX','MRK','DIS','NKE','MMM','KO','WMT','DOW','CSCO','INTC','VZ']
NDX_100 = ['MSFT','AAPL','AMZN','META','AVGO','GOOGL','GOOG','COST','TSLA','NFLX','AMD','PEP','QCOM','TMUS','ADBE','LIN','CSCO','AMAT','TXN','AMGN','INTU','CMCSA','ISRG','MU','HON','BKNG','INTC','LRCX','VRTX','NVDA','ADI','REGN','KLAC','ADP','PANW','PDD','SBUX','MDLZ','ASML','SNPS','MELI','GILD','CDNS','CRWD','PYPL','NXPI','CTAS','MAR','ABNB','CSX','CEG','ROP','MRVL','ORLY','MRNA','PCAR','MNST','CPRT','MCHP','ROST','KDP','AZN','AEP','ADSK','FTNT','WDAY','DXCM','PAYX','DASH','TTD','KHC','IDXX','CHTR','LULU','VRSK','ODFL','EA','FAST','EXC','GEHC','CCEP','FANG','DDOG','CTSH','BIIB','BKR','CSGP','ON','XEL','CDW','ANSS','TTWO','ZS','GFS','TEAM','DLTR','WBD','ILMN','MDB','WBA','SIRI']
SNP_500 = ['MSFT','AAPL','NVDA','AMZN','META','GOOGL','GOOG','BRK.B','LLY','AVGO','JPM','XOM','TSLA','UNH','V','PG','MA','JNJ','COST','MRK','HD','ABBV','WMT','NFLX','CVX','CRM','BAC','AMD','PEP','KO','TMO','QCOM','ADBE','WFC','LIN','ORCL','ACN','CSCO','MCD','INTU','DIS','AMAT','ABT','GE','TXN','CAT','DHR','VZ','AMGN','PFE','IBM','NOW','PM','NEE','CMCSA','GS','UNP','SPGI','ISRG','RTX','MU','COP','ETN','AXP','HON','BKNG','UBER','ELV','INTC','LRCX','LOW','T','MS','C','PGR','ADI','VRTX','TJX','SYK','NKE','BLK','BSX','MDT','SCHW','CB','REGN','KLAC','ADP','MMC','UPS','LMT','CI','BA','DE','PANW','PLD','MDLZ','FI','SNPS','SBUX','BX','AMT','TMUS','CMG','BMY','SO','GILD','APH','MO','CDNS','ZTS','DUK','ICE','CL','WM','CME','TT','ANET','TDG','FCX','MCK','EOG','EQIX','SHW','CEG','NXPI','CVS','PH','GD','BDX','TGT','CSX','PYPL','SLB','NOC','ITW','MPC','EMR','MCO','HCA','USB','ABNB','PSX','PNC','MSI','CTAS','ECL','APD','ROP','ORLY','FDX','MAR','PCAR','AON','WELL','VLO','MMM','AIG','MRNA','AJG','CARR','MCHP','EW','COF','NSC','TFC','GM','HLT','JCI','WMB','DXCM','TRV','AZO','SRE','NEM','F','SPG','AEP','OKE','CPRT','TEL','ADSK','DLR','AFL','FIS','URI','ROST','KMB','BK','A','MET','GEV','D','AMP','PSA','O','HUM','CCI','ALL','IDXX','SMCI','DHI','PRU','LHX','NUE','GWW','IQV','HES','CNC','OXY','PAYX','PWR','DOW','AME','OTIS','STZ','GIS','PCG','CTVA','MNST','FTNT','MSCI','CMI','IR','YUM','LEN','RSG','ACGL','FAST','KVUE','KMI','EXC','PEG','COR','SYY','VRSK','MPWR','MLM','KDP','CSGP','KR','IT','RCL','LULU','XYL','FANG','CTSH','VMC','DD','GEHC','FICO','DAL','EA','ED','ADM','VST','HWM','HAL','MTD','BKR','BIIB','RMD','CDW','DVN','PPG','ON','DFS','ODFL','DG','TSCO','WAB','HIG','HSY','EXR','ROK','XEL','VICI','EL','EFX','ANSS','KHC','EIX','HPQ','GLW','EBAY','AVB','GPN','FSLR','FTV','CHTR','TROW','CBRE','CHD','WTW','DOV','WEC','TRGP','KEYS','FITB','GRMN','AWK','MTB','LYB','WST','ZBH','IFF','TTWO','DLTR','PHM','WDC','BR','NTAP','CAH','NVR','HPE','NDAQ','RJF','DECK','ETR','IRM','DTE','STT','STE','APTV','EQR','WY','VLTO','PTC','BALL','HUBB','TER','PPL','BRO','BLDR','TYL','GPC','LDOS','SBAC','CTRA','STLD','FE','ES','WAT','INVH','MOH','AXON','CPAY','HBAN','CBOE','VTR','TDY','COO','OMC','CNP','AEE','ARE','ULTA','CINF','AVY','NRG','MKC','STX','ALGN','PFG','CMS','DRI','SYF','RF','DPZ','J','HOLX','TSN','BAX','TXT','NTRS','WBD','UAL','EXPD','ATO','EG','ILMN','LH','ZBRA','FDS','ESS','LVS','EQT','CFG','CLX','IEX','K','PKG','WRB','ENPH','LUV','DGX','MAA','IP','VRSN','JBL','MAS','CE','MRO','CF','CCL','BG','EXPE','SWKS','CAG','ALB','SNA','AMCR','AKAM','TRMB','POOL','GEN','RVTY','AES','L','PNR','BBY','DOC','WRK','KEY','LYV','SWK','JBHT','NDSN','HST','ROL','TECH','VTRS','LNT','LW','KIM','EVRG','JKHY','IPG','PODD','UDR','EMN','WBA','LKQ','NI','SJM','CPT','CRL','JNPR','BBWI','KMX','UHS','EPAM','INCY','ALLE','MGM','AOS','MOS','FFIV','HII','HRL','TAP','CTLT','NWSA','CHRW','REG','TFX','TPR','QRVO','HSIC','DAY','APA','WYNN','AAL','CPB','GNRC','AIZ','PNW','PAYC','BXP','BF.B','SOLV','BWA','MKTX','FOXA','MTCH','HAS','FMC','ETSY','FRT','DVA','RHI','IVZ','CZR','GL','RL','CMA','BEN','NCLH','BIO','MHK','PARA','FOX','NWS']


def check_mid(x):
    return isinstance(x, int|float) and (0 < x < float('inf'))

def it_check_mid(x):
    return list(map(check_mid, x))

def np_check_mid(x, flt_warn=True):
    if flt_warn:  warnings.filterwarnings('ignore', 'invalid value encountered')
    return np.vectorize(check_mid)(x)


def calc_mid(ask, bid, alt=float('nan')):

    A, B = check_mid(ask), check_mid(bid)

    if A and B: return (ask * bid)**(1/2)
    if A:       return  ask 
    if B:       return  bid 
    else:       return  alt 

def np_calc_mid(ask, bid, alt=np.nan, flt_warn=True):
    A  = np_check_mid(ask, flt_warn) 
    B  = np_check_mid(bid, flt_warn) 
    AB = np.logical_and(A, B)
    return np.select([AB, A, B], [np.sqrt(ask*bid), ask, bid], alt)

def pd_calc_mid(Df, Ask, Bid, alt=float('nan')):
    def Lambda(x):
        ask, bid = x[Ask], x[Bid]
        A, B = check_mid(ask), check_mid(bid)
        if A and B: return (ask * bid)**(1/2)
        if A:       return  ask 
        if B:       return  bid 
        else:       return  alt 

    if isinstance(Df, pd.DataFrame):    return Df.apply(Lambda, axis=1)
    else:                               return Lambda



# =============================================== #
# ================ FORMAT SOURCE ================ #
# =============================================== #
def FORMAT_OPT_CHAIN(Src, TOP=10, RankBy:Lit['Liq','Notion']='Liq', OI_base=1000, IV_coef=100, DTY=365.25, 
    N='N', TICKER='Ticker', PRICE='Price', STRIKE='Strike', EXPDATE='ExpDate', UPDATE='Update', 
    CALL_CODE='Call Code', CALL_VOL='Call Vol', CALL_OI='Call OI', CALL_ASK='Call Ask', CALL_BID='Call Bid', CALL_LAST_PRICE='Call Last Price', CALL_IV='Call IV', 
     PUT_CODE= 'Put Code',  PUT_VOL= 'Put Vol',  PUT_OI= 'Put OI',  PUT_ASK= 'Put Ask',  PUT_BID= 'Put Bid',  PUT_LAST_PRICE= 'Put Last Price',  PUT_IV= 'Put IV', 
):

    # ================ Helpers ================ # 
    def _spd_ratio(Df, Y, X, rnd=2, alt=nan):

        def Lambda(row):
            dy, dx = row[Y], row[X]
            if dy and dx:
                try:    return round(dy/dx *100-100, rnd)
                except: return alt
            else:       return alt

        if isinstance(Df, pd.DataFrame): 
                return Df.apply(Lambda, axis=1)
        else:   return Lambda


    # ================ Main ================ # 
    np_Today = np.datetime64( dt.date.today(), 'D')
    pd_Today = pd.to_datetime(dt.date.today())
    
    Tmp                 = pd.DataFrame()
    Tmp['N']            = Src[N]
    Tmp['Ticker']       = Src[TICKER]


    Tmp['Call Liq']     = Src[CALL_VOL].fillna(0) + Src[CALL_OI].fillna(0)/OI_base
    Tmp['Put Liq']      = Src[PUT_VOL].fillna(0)  + Src[PUT_OI].fillna(0)/OI_base

    Tmp['Call Mid']     = np_calc_mid(Src[CALL_ASK], Src[CALL_BID])
    Tmp['Put Mid']      = np_calc_mid( Src[PUT_ASK],  Src[PUT_BID])
    
    Tmp['Call Last']    = Src[CALL_LAST_PRICE]
    Tmp['Put Last']     = Src[PUT_LAST_PRICE]


    Tmp['Call Notion']  = Tmp['Call Liq'] * pd_num_coalesce(Tmp, ['Call Mid', 'Call Last'], check='is_geo', logErr=0)
    Tmp['Put Notion']   = Tmp['Put Liq']  * pd_num_coalesce(Tmp, [ 'Put Mid',  'Put Last'], check='is_geo', logErr=0)

    Tmp['Call Rank']    = Tmp.groupby(['Ticker', 'N'])[f'Call {RankBy}'].rank(ascending=0).apply(py_int)
    Tmp['Put Rank']     = Tmp.groupby(['Ticker', 'N'])[f'Put {RankBy}'].rank(ascending=0).apply(py_int)

    Tmp['Call InTop']   = (Tmp['Call Rank'] <= TOP) *1
    Tmp['Put InTop']    = (Tmp['Put Rank']  <= TOP) *1
    Tmp['InTop']        =  np.bitwise_or(Tmp['Call InTop'], Tmp['Put InTop']) 


    # ===================================================== #
    # ======================== IDs ======================== #
    # ===================================================== #
    Tmp['Price']    = Src[PRICE].astype(float)
    
    Df              =  pd.DataFrame()
    Df['A']         =  Src[N]
    Df['Z']         = -Src.groupby(TICKER)[N].rank(method='dense', ascending=0).astype(int)
    Df['Ticker']    =  Src[TICKER]
    Df['Price']     =  Tmp['Price']
    Df['Update']    = pd_getter(Src, UPDATE, _as='date', err_rtn='nat')  
    Df['InTop']     =  Tmp['InTop']
    
    # ====================================================== #
    # ======================== CALL ======================== #
    # ====================================================== #
    Df['CALL']      = ''
    Df['Call Code'] = pd_getter(Src, CALL_CODE, _as='same', err_rtn='')
    Df['Call Code'] = Df['Call Code'].fillna('')
    Df['Call InTop']= Tmp['Call InTop']
    Df['Call Rank'] = Tmp['Call Rank']
    
    Df['Call Notion']= Tmp['Call Notion'].round(2)
    Df['Call Liq']  = Tmp['Call Liq']
    Df['Call Vol']  = Src[CALL_VOL]
    Df['Call OI']   = Src[CALL_OI]

    Df['Call Ask']  = Src[CALL_ASK]
    Df['Call Mid']  = Tmp['Call Mid'].round(2)
    Df['Call Bid']  = Src[CALL_BID]
    Df['Call Last'] = Tmp['Call Last']
    Df['Call A/B']  = _spd_ratio(Src, Y=CALL_ASK, X=CALL_BID)
    Df['Call B/A']  = _spd_ratio(Src, Y=CALL_BID, X=CALL_ASK)
    Df['Call IV']   = pd_getter(Src, CALL_IV, _as='same', err_rtn='nan', mult=IV_coef, fig=2)

    # ====================================================== #
    # ======================== AXIS ======================== #
    # ====================================================== #
    Tmp['Chg']      =  Src[STRIKE] - Tmp['Price']
    Tmp['Var']      = (Src[STRIKE] / Tmp['Price'] *100-100).round(2)
    Tmp['Side']     = np.sign(Tmp['Chg']).astype(int)
    Tmp['S']        = (
        (Tmp['Side'] > 0) * (+Tmp.groupby(['Ticker', 'N', 'Side'])['Chg'].rank(ascending=1).astype(int)) + 
        (Tmp['Side'] < 0) * (-Tmp.groupby(['Ticker', 'N', 'Side'])['Chg'].rank(ascending=0).astype(int))
    )
    Tmp['K']        = (
        (Tmp['InTop']==1) * (Tmp['Side'] > 0) * (+Tmp.groupby(['Ticker', 'N', 'InTop', 'Side'])['Chg'].rank(ascending=1).astype(int)) + 
        (Tmp['InTop']==1) * (Tmp['Side'] < 0) * (-Tmp.groupby(['Ticker', 'N', 'InTop', 'Side'])['Chg'].rank(ascending=0).astype(int))
    )

    Df['AXIS']      = ''
    Df['ExpDate']   =   Src[EXPDATE]
    Df['YTE']       = ((Src[EXPDATE] - pd_Today).dt.days /DTY    ).round(2)
    Df['QTE']       = ((Src[EXPDATE] - pd_Today).dt.days /DTY * 4).round(2)
    Df['MTE']       = ((Src[EXPDATE] - pd_Today).dt.days /DTY *12).round(2)
    Df['WTE']       = ((Src[EXPDATE] - pd_Today).dt.days /7      ).round(2)
    Df['DTE']       =   Src[EXPDATE].apply(lambda x: np.busday_count(np_Today, np.datetime64(x, 'D')))
    
    Df['Side']      = Tmp['Side']
    Df['S']         = Tmp['S']
    Df['K']         = Tmp['K']
    Df['Strike']    = Src[STRIKE]
    Df['Chg']       = Tmp['Chg']
    Df['Var']       = Tmp['Var']
    
    # ===================================================== #
    # ======================== PUT ======================== #
    # ===================================================== #
    Df['PUT']       = ''
    Df['Put Code']  = pd_getter(Src, PUT_CODE, _as='same', err_rtn='')
    Df['Put Code']  = Df['Put Code'].fillna('')
    Df['Put InTop'] = Tmp['Put InTop'] 
    Df['Put Rank']  = Tmp['Put Rank']
    
    Df['Put Notion']= Tmp['Put Notion'].round(2)
    Df['Put Liq']   = Tmp['Put Liq']
    Df['Put Vol']   = Src[PUT_VOL]
    Df['Put OI']    = Src[PUT_OI]

    Df['Put Ask']   = Src[PUT_ASK]
    Df['Put Mid']   = Tmp['Put Mid'].round(2)
    Df['Put Bid']   = Src[PUT_BID]
    Df['Put A/B']   = _spd_ratio(Src, Y=PUT_ASK, X=PUT_BID)
    Df['Put B/A']   = _spd_ratio(Src, Y=PUT_BID, X=PUT_ASK)

    Df['Put Last']  = Tmp['Put Last']
    Df['Put IV']    = pd_getter(Src, PUT_IV, _as='same', err_rtn='nan', mult=IV_coef, fig=2)
    
    return Df














# =========================================================================================== #
# ======================================== PROVIDERS ======================================== #
# =========================================================================================== #

# ================ MW OPTIONS ================ #
def MW_OptChain(TICKER='SPY', showAll=False, 
    tries=3, catch=True, err_rtn:Lit['Empty','none','void','alt']='Empty', alt=None, logErr=True,
):
    while tries:
        try: 
            # ======================================================================== #
            # ================================ Step 1 ================================ #
            # ======================================================================== #
            def URL(TICKER, showAll=False):
                return f'https://bigcharts.marketwatch.com/quickchart/options.asp?symb={TICKER}&showAll={showAll}'

            endpoint = URL(TICKER, showAll)
            # endpoint

            def SCAN_PAGE():
                Page_Tables = pd.read_html(endpoint)
                Raw = Page_Tables[2]
                return Raw

            Raw = SCAN_PAGE()
            # Raw

            # ======================================================================== #
            # ================================ Step 2 ================================ #
            # ======================================================================== #
            def META(Raw):
                Prices, Updates, ExpDate, Heads, Centers = [], [], [], [], []

                for idx, row in Raw.iterrows():
                    
                    if isinstance(row[6], str) and row[6].startswith('Expires'):
                        ExpDate.append(dt.datetime.strptime(row[6].replace('Expires ', ''), '%B %d, %Y'))
                    
                    if (row[6] == 'StrikePrice'):
                        Heads.append(idx)

                    if (row[5] == 'Stock Price »'):
                        Centers.append(idx)
                        Prices.append(row[6])
                        Updates.append(dt.datetime.strptime(row[7].replace('Last as of ', ''), '%m/%d/%Y %I:%M:%S %p'))
                pass
                Meta              = pd.DataFrame({ 'Price':Prices, 'Update':Updates, 'ExpDate':ExpDate, 'Head':Heads, 'Center':Centers })
                Meta.insert(0, 'N', Meta.index)
                Meta['Tail']      = Meta['Head'].shift(-1).astype('Int64')
                return Meta

            Meta = META(Raw)
            # Meta

            # ======================================================================== #
            # ================================ Step 3 ================================ #
            # ======================================================================== #
            def SPLIT(Raw, Meta):
                Secs = []
                for i, meta in Meta.iterrows():
                    H   = meta['Head']
                    C   = meta['Center']
                    T   = meta['Tail'] if pd.notna(meta['Tail']) else None
                    Sec = Raw.iloc[H:T].drop(C).reset_index(drop=1)
                    Secs.append(Sec)
                pass
                return Secs

            Secs = SPLIT(Raw, Meta)
            # Secs[0]

            # ======================================================================== #
            # ================================ Step 4 ================================ #
            # ======================================================================== #
            def FRAME(Secs):
                Frames = []
                for Sec in Secs:
                    Frame = Sec.iloc[:, None:-2]
                    Frame.columns = [
                        'Call Last Price', 'Call Chg', 'Call Vol', 'Call Bid', 'Call Ask', 'Call OI', 'Strike', 
                        'Put Last Price',  'Put Chg',  'Put Vol',  'Put Bid',  'Put Ask',  'Put OI'
                    ]
                    is_num = pd.notna(pd.to_numeric(Frame['Strike'], 'coerce'))
                    Frame  = Frame[is_num].reset_index(drop=1)
                    Frames.append(Frame)
                pass
                return Frames

            Frames = FRAME(Secs)
            # Frames[0]

            # ======================================================================== #
            # ================================ Step 5 ================================ #
            # ======================================================================== #
            def SOURCE(Frames, Meta, TICKER=''):
                Sources = []
                for N, Frame in enumerate(Frames):
                    Src = Frame.astype(float)
                    Src = Src.sort_values('Strike', ascending=0).reset_index(drop=1)
                    Src = Src[[
                        'Call Vol', 'Call OI', 'Call Ask', 'Call Bid', 'Call Last Price', 'Call Chg', 'Strike',
                        'Put Vol',  'Put OI',  'Put Ask',  'Put Bid',  'Put Last Price',  'Put Chg',
                    ]]
                    Src.insert(0, 'N'      , N                     )
                    Src.insert(1, 'Ticker' , TICKER                )
                    Src.insert(2, 'Price'  , Meta.loc[N, 'Price']  )
                    Src.insert(3, 'Update' , Meta.loc[N, 'Update'] )
                    Src.insert(4, 'ExpDate', Meta.loc[N, 'ExpDate'])
                    Sources.append(Src)
                pass
                return pd.concat(Sources) .reset_index(drop=1)

            Src = SOURCE(Frames, Meta, TICKER)
            # Src

            return Src


        except Exception as Error:
            if not catch: raise Error
            if logErr: print('Ticker:',TICKER, 'Tries:',tries, 'Error:',Error)
            tries -= 1

        if not tries:
            Empty = pd.DataFrame({ 
                'N', 'Ticker', 'Price', 'Update', 'ExpDate', 
                'Call Vol', 'Call OI', 'Call Ask', 'Call Bid', 'Call Last Price', 'Call Chg', 
                'Strike', 
                'Put Vol', 'Put OI', 'Put Ask', 'Put Bid', 'Put Last Price', 'Put Chg',
            })

            if err_rtn == 'Empty':  return Empty
            if err_rtn == 'none':   return None
            if err_rtn == 'alt':    return alt

def MW_OptChains(TICKERS, showAll=False, 
    tries=3, catch=True, err_rtn:Lit['Empty','none','void','alt']='Empty', alt=None, logErr=True, verbose=True, 
):
    return pd_reduce_Secs(np.ravel(TICKERS), lambda i, TICKER: (
        MW_OptChain(TICKER, showAll, tries, catch, err_rtn, alt, logErr)        
    ), Initial=[], verbose=verbose)




