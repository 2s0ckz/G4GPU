// NIST material compositions, extracted from Geant4 11.1.1 G4NistManager by
// ref/dump/g4dump.cc. Do not edit by hand; re-run the oracle instead.
//
// Each entry is a density in g/cm3, a mean excitation energy in eV, a state,
// the tabulated Sternheimer density-effect parameters where Geant4 has them,
// and the element mass fractions.
#pragma once
#include "g4/G4Types.hh"

namespace g4gpu::g4::nist {

struct NistComponent { int z; double fraction; };
struct NistMaterial {
  const char* name;
  double density_g_cm3;
  double mean_excitation_eV;
  int state;            ///< 0 undefined, 1 solid, 2 liquid, 3 gas
  bool has_sternheimer;
  double cbar, x0, x1, a, m, delta0;
  int n_components;
  const NistComponent* components;
};

inline constexpr NistComponent c_G4_WATER[] = {{1, 0.11189847784106703}, {8, 0.8881015221589329}};
inline constexpr NistComponent c_G4_H[] = {{1, 1}};
inline constexpr NistComponent c_G4_He[] = {{2, 1}};
inline constexpr NistComponent c_G4_Li[] = {{3, 1}};
inline constexpr NistComponent c_G4_Be[] = {{4, 1}};
inline constexpr NistComponent c_G4_B[] = {{5, 1}};
inline constexpr NistComponent c_G4_C[] = {{6, 1}};
inline constexpr NistComponent c_G4_N[] = {{7, 1}};
inline constexpr NistComponent c_G4_O[] = {{8, 1}};
inline constexpr NistComponent c_G4_F[] = {{9, 1}};
inline constexpr NistComponent c_G4_Ne[] = {{10, 1}};
inline constexpr NistComponent c_G4_Na[] = {{11, 1}};
inline constexpr NistComponent c_G4_Mg[] = {{12, 1}};
inline constexpr NistComponent c_G4_Al[] = {{13, 1}};
inline constexpr NistComponent c_G4_Si[] = {{14, 1}};
inline constexpr NistComponent c_G4_P[] = {{15, 1}};
inline constexpr NistComponent c_G4_S[] = {{16, 1}};
inline constexpr NistComponent c_G4_Cl[] = {{17, 1}};
inline constexpr NistComponent c_G4_Ar[] = {{18, 1}};
inline constexpr NistComponent c_G4_K[] = {{19, 1}};
inline constexpr NistComponent c_G4_Ca[] = {{20, 1}};
inline constexpr NistComponent c_G4_Sc[] = {{21, 1}};
inline constexpr NistComponent c_G4_Ti[] = {{22, 1}};
inline constexpr NistComponent c_G4_V[] = {{23, 1}};
inline constexpr NistComponent c_G4_Cr[] = {{24, 1}};
inline constexpr NistComponent c_G4_Mn[] = {{25, 1}};
inline constexpr NistComponent c_G4_Fe[] = {{26, 1}};
inline constexpr NistComponent c_G4_Co[] = {{27, 1}};
inline constexpr NistComponent c_G4_Ni[] = {{28, 1}};
inline constexpr NistComponent c_G4_Cu[] = {{29, 1}};
inline constexpr NistComponent c_G4_Zn[] = {{30, 1}};
inline constexpr NistComponent c_G4_Ga[] = {{31, 1}};
inline constexpr NistComponent c_G4_Ge[] = {{32, 1}};
inline constexpr NistComponent c_G4_As[] = {{33, 1}};
inline constexpr NistComponent c_G4_Se[] = {{34, 1}};
inline constexpr NistComponent c_G4_Br[] = {{35, 1}};
inline constexpr NistComponent c_G4_Kr[] = {{36, 1}};
inline constexpr NistComponent c_G4_Rb[] = {{37, 1}};
inline constexpr NistComponent c_G4_Sr[] = {{38, 1}};
inline constexpr NistComponent c_G4_Y[] = {{39, 1}};
inline constexpr NistComponent c_G4_Zr[] = {{40, 1}};
inline constexpr NistComponent c_G4_Nb[] = {{41, 1}};
inline constexpr NistComponent c_G4_Mo[] = {{42, 1}};
inline constexpr NistComponent c_G4_Tc[] = {{43, 1}};
inline constexpr NistComponent c_G4_Ru[] = {{44, 1}};
inline constexpr NistComponent c_G4_Rh[] = {{45, 1}};
inline constexpr NistComponent c_G4_Pd[] = {{46, 1}};
inline constexpr NistComponent c_G4_Ag[] = {{47, 1}};
inline constexpr NistComponent c_G4_Cd[] = {{48, 1}};
inline constexpr NistComponent c_G4_In[] = {{49, 1}};
inline constexpr NistComponent c_G4_Sn[] = {{50, 1}};
inline constexpr NistComponent c_G4_Sb[] = {{51, 1}};
inline constexpr NistComponent c_G4_Te[] = {{52, 1}};
inline constexpr NistComponent c_G4_I[] = {{53, 1}};
inline constexpr NistComponent c_G4_Xe[] = {{54, 1}};
inline constexpr NistComponent c_G4_Cs[] = {{55, 1}};
inline constexpr NistComponent c_G4_Ba[] = {{56, 1}};
inline constexpr NistComponent c_G4_La[] = {{57, 1}};
inline constexpr NistComponent c_G4_Ce[] = {{58, 1}};
inline constexpr NistComponent c_G4_Pr[] = {{59, 1}};
inline constexpr NistComponent c_G4_Nd[] = {{60, 1}};
inline constexpr NistComponent c_G4_Pm[] = {{61, 1}};
inline constexpr NistComponent c_G4_Sm[] = {{62, 1}};
inline constexpr NistComponent c_G4_Eu[] = {{63, 1}};
inline constexpr NistComponent c_G4_Gd[] = {{64, 1}};
inline constexpr NistComponent c_G4_Tb[] = {{65, 1}};
inline constexpr NistComponent c_G4_Dy[] = {{66, 1}};
inline constexpr NistComponent c_G4_Ho[] = {{67, 1}};
inline constexpr NistComponent c_G4_Er[] = {{68, 1}};
inline constexpr NistComponent c_G4_Tm[] = {{69, 1}};
inline constexpr NistComponent c_G4_Yb[] = {{70, 1}};
inline constexpr NistComponent c_G4_Lu[] = {{71, 1}};
inline constexpr NistComponent c_G4_Hf[] = {{72, 1}};
inline constexpr NistComponent c_G4_Ta[] = {{73, 1}};
inline constexpr NistComponent c_G4_W[] = {{74, 1}};
inline constexpr NistComponent c_G4_Re[] = {{75, 1}};
inline constexpr NistComponent c_G4_Os[] = {{76, 1}};
inline constexpr NistComponent c_G4_Ir[] = {{77, 1}};
inline constexpr NistComponent c_G4_Pt[] = {{78, 1}};
inline constexpr NistComponent c_G4_Au[] = {{79, 1}};
inline constexpr NistComponent c_G4_Hg[] = {{80, 1}};
inline constexpr NistComponent c_G4_Tl[] = {{81, 1}};
inline constexpr NistComponent c_G4_Pb[] = {{82, 1}};
inline constexpr NistComponent c_G4_Bi[] = {{83, 1}};
inline constexpr NistComponent c_G4_Po[] = {{84, 1}};
inline constexpr NistComponent c_G4_At[] = {{85, 1}};
inline constexpr NistComponent c_G4_Rn[] = {{86, 1}};
inline constexpr NistComponent c_G4_Fr[] = {{87, 1}};
inline constexpr NistComponent c_G4_Ra[] = {{88, 1}};
inline constexpr NistComponent c_G4_Ac[] = {{89, 1}};
inline constexpr NistComponent c_G4_Th[] = {{90, 1}};
inline constexpr NistComponent c_G4_Pa[] = {{91, 1}};
inline constexpr NistComponent c_G4_U[] = {{92, 1}};
inline constexpr NistComponent c_G4_Np[] = {{93, 1}};
inline constexpr NistComponent c_G4_Pu[] = {{94, 1}};
inline constexpr NistComponent c_G4_Am[] = {{95, 1}};
inline constexpr NistComponent c_G4_Cm[] = {{96, 1}};
inline constexpr NistComponent c_G4_Bk[] = {{97, 1}};
inline constexpr NistComponent c_G4_Cf[] = {{98, 1}};
inline constexpr NistComponent c_G4_A_150_TISSUE[] = {{1, 0.10132689867310134}, {6, 0.77550022449977551}, {7, 0.035056964943035056}, {8, 0.052315947684052323}, {9, 0.017421982578017425}, {20, 0.018377981622018379}};
inline constexpr NistComponent c_G4_ACETONE[] = {{6, 0.62039735053092393}, {1, 0.10412746607058551}, {8, 0.27547518339849059}};
inline constexpr NistComponent c_G4_ACETYLENE[] = {{6, 0.9225773292943783}, {1, 0.077422670705621713}};
inline constexpr NistComponent c_G4_ADENINE[] = {{6, 0.44442324240708031}, {1, 0.03729598946152831}, {7, 0.51828076813139135}};
inline constexpr NistComponent c_G4_ADIPOSE_TISSUE_ICRP[] = {{1, 0.114}, {6, 0.59799999999999998}, {7, 0.0070000000000000001}, {8, 0.27800000000000002}, {11, 0.001}, {16, 0.001}, {17, 0.001}};
inline constexpr NistComponent c_G4_AIR[] = {{6, 0.000124000124000124}, {7, 0.75526775526775525}, {8, 0.23178123178123175}, {18, 0.012827012827012825}};
inline constexpr NistComponent c_G4_ALANINE[] = {{6, 0.40443210957366915}, {1, 0.079193180284683556}, {7, 0.15721453815928274}, {8, 0.3591601719823645}};
inline constexpr NistComponent c_G4_ALUMINUM_OXIDE[] = {{13, 0.52925049160807136}, {8, 0.47074950839192858}};
inline constexpr NistComponent c_G4_AMBER[] = {{1, 0.10593010593010593}, {6, 0.788973788973789}, {8, 0.10509610509610509}};
inline constexpr NistComponent c_G4_AMMONIA[] = {{7, 0.82244760506022285}, {1, 0.17755239493977706}};
inline constexpr NistComponent c_G4_ANILINE[] = {{6, 0.77383137350005049}, {1, 0.075763231974996001}, {7, 0.15040539452495352}};
inline constexpr NistComponent c_G4_ANTHRACENE[] = {{6, 0.94344709903049606}, {1, 0.056552900969503946}};
inline constexpr NistComponent c_G4_B_100_BONE[] = {{1, 0.065470934529065467}, {6, 0.53694446305553689}, {7, 0.021499978500021496}, {8, 0.032084967915032084}, {9, 0.16741083258916739}, {20, 0.17658882341117657}};
inline constexpr NistComponent c_G4_BAKELITE[] = {{1, 0.057440999999999999}, {6, 0.77459100000000003}, {8, 0.16796800000000001}};
inline constexpr NistComponent c_G4_BARIUM_FLUORIDE[] = {{56, 0.78327618100639806}, {9, 0.21672381899360199}};
inline constexpr NistComponent c_G4_BARIUM_SULFATE[] = {{56, 0.58839933026387015}, {16, 0.13739255739537751}, {8, 0.27420811234075232}};
inline constexpr NistComponent c_G4_BENZENE[] = {{6, 0.9225773292943783}, {1, 0.077422670705621713}};
inline constexpr NistComponent c_G4_BERYLLIUM_OXIDE[] = {{4, 0.36032043777772843}, {8, 0.6396795622222714}};
inline constexpr NistComponent c_G4_BGO[] = {{83, 0.67101689612131676}, {32, 0.1748650835883821}, {8, 0.15411802029030117}};
inline constexpr NistComponent c_G4_BLOOD_ICRP[] = {{1, 0.10199999999999999}, {6, 0.11}, {7, 0.033000000000000002}, {8, 0.745}, {11, 0.001}, {15, 0.001}, {16, 0.002}, {17, 0.0030000000000000001}, {19, 0.002}, {26, 0.001}};
inline constexpr NistComponent c_G4_BONE_COMPACT_ICRU[] = {{1, 0.064000000000000001}, {6, 0.27800000000000002}, {7, 0.027}, {8, 0.40999999999999998}, {12, 0.002}, {15, 0.070000000000000007}, {16, 0.002}, {20, 0.14699999999999999}};
inline constexpr NistComponent c_G4_BONE_CORTICAL_ICRP[] = {{1, 0.034000000000000002}, {6, 0.155}, {7, 0.042000000000000003}, {8, 0.435}, {11, 0.001}, {12, 0.002}, {15, 0.10299999999999999}, {16, 0.0030000000000000001}, {20, 0.22500000000000001}};
inline constexpr NistComponent c_G4_BORON_CARBIDE[] = {{5, 0.7826299986678551}, {6, 0.21737000133214482}};
inline constexpr NistComponent c_G4_BORON_OXIDE[] = {{5, 0.3105712357543674}, {8, 0.6894287642456326}};
inline constexpr NistComponent c_G4_BRAIN_ICRP[] = {{1, 0.107}, {6, 0.14499999999999999}, {7, 0.021999999999999999}, {8, 0.71199999999999997}, {11, 0.002}, {15, 0.0040000000000000001}, {16, 0.002}, {17, 0.0030000000000000001}, {19, 0.0030000000000000001}};
inline constexpr NistComponent c_G4_BUTANE[] = {{6, 0.82658294101477181}, {1, 0.17341705898522822}};
inline constexpr NistComponent c_G4_N_BUTYL_ALCOHOL[] = {{6, 0.64816264808245883}, {1, 0.13598449060241222}, {8, 0.21585286131512904}};
inline constexpr NistComponent c_G4_C_552[] = {{1, 0.024680024680024681}, {6, 0.50161050161050158}, {8, 0.0045270045270045271}, {9, 0.46520946520946521}, {14, 0.0039730039730039727}};
inline constexpr NistComponent c_G4_CADMIUM_TELLURIDE[] = {{48, 0.46835318560669537}, {52, 0.53164681439330452}};
inline constexpr NistComponent c_G4_CADMIUM_TUNGSTATE[] = {{48, 0.31203678986130257}, {74, 0.51031587675873502}, {8, 0.1776473333799625}};
inline constexpr NistComponent c_G4_CALCIUM_CARBONATE[] = {{20, 0.40043218365675287}, {6, 0.12000303405741311}, {8, 0.47956478228583393}};
inline constexpr NistComponent c_G4_CALCIUM_FLUORIDE[] = {{20, 0.513328441427641}, {9, 0.486671558572359}};
inline constexpr NistComponent c_G4_CALCIUM_OXIDE[] = {{20, 0.71469104988164511}, {8, 0.28530895011835483}};
inline constexpr NistComponent c_G4_CALCIUM_SULFATE[] = {{20, 0.29438466994218276}, {16, 0.23553483233197239}, {8, 0.47008049772584476}};
inline constexpr NistComponent c_G4_CALCIUM_TUNGSTATE[] = {{20, 0.1391998504560634}, {74, 0.63852249154428298}, {8, 0.22227765799965371}};
inline constexpr NistComponent c_G4_CARBON_DIOXIDE[] = {{6, 0.27291225043146294}, {8, 0.72708774956853706}};
inline constexpr NistComponent c_G4_CARBON_TETRACHLORIDE[] = {{6, 0.078082537748977832}, {17, 0.92191746225102222}};
inline constexpr NistComponent c_G4_CELLULOSE_CELLOPHANE[] = {{6, 0.44445585638851354}, {1, 0.062164544043644493}, {8, 0.49337959956784205}};
inline constexpr NistComponent c_G4_CELLULOSE_BUTYRATE[] = {{1, 0.067125000000000004}, {6, 0.54540299999999997}, {8, 0.38747199999999998}};
inline constexpr NistComponent c_G4_CELLULOSE_NITRATE[] = {{1, 0.029215999999999999}, {6, 0.27129599999999998}, {7, 0.12127599999999999}, {8, 0.57821199999999995}};
inline constexpr NistComponent c_G4_CERIC_SULFATE[] = {{1, 0.10759600000000001}, {7, 0.00080000000000000015}, {8, 0.87497600000000009}, {16, 0.014627000000000001}, {58, 0.0020010000000000006}};
inline constexpr NistComponent c_G4_CESIUM_FLUORIDE[] = {{55, 0.87493104170150238}, {9, 0.1250689582984976}};
inline constexpr NistComponent c_G4_CESIUM_IODIDE[] = {{55, 0.51154886859192716}, {53, 0.48845113140807284}};
inline constexpr NistComponent c_G4_CHLOROBENZENE[] = {{6, 0.64024994689263703}, {1, 0.044774802098408117}, {17, 0.31497525100895485}};
inline constexpr NistComponent c_G4_CHLOROFORM[] = {{6, 0.10061232076328948}, {1, 0.0084433839116146315}, {17, 0.89094429532509589}};
inline constexpr NistComponent c_G4_CONCRETE[] = {{1, 0.01}, {6, 0.001}, {8, 0.52910699999999999}, {11, 0.016}, {12, 0.002}, {13, 0.033871999999999999}, {14, 0.33702100000000002}, {19, 0.012999999999999999}, {20, 0.043999999999999997}, {26, 0.014}};
inline constexpr NistComponent c_G4_CYCLOHEXANE[] = {{6, 0.85628171225519811}, {1, 0.14371828774480183}};
inline constexpr NistComponent c_G4_1_2_DICHLOROBENZENE[] = {{6, 0.49022970891497997}, {1, 0.027426711465994656}, {17, 0.4823435796190253}};
inline constexpr NistComponent c_G4_DICHLORODIETHYL_ETHER[] = {{6, 0.33593879197932897}, {1, 0.056383953177242728}, {8, 0.11187523639322854}, {17, 0.49580201845019972}};
inline constexpr NistComponent c_G4_1_2_DICHLOROETHANE[] = {{6, 0.24274318290163677}, {1, 0.040742005941555373}, {17, 0.71651481115680782}};
inline constexpr NistComponent c_G4_DIETHYL_ETHER[] = {{6, 0.64816264808245883}, {1, 0.13598449060241222}, {8, 0.21585286131512904}};
inline constexpr NistComponent c_G4_N_N_DIMETHYL_FORMAMIDE[] = {{6, 0.49295745101736077}, {1, 0.096527618275039084}, {7, 0.19162691625936451}, {8, 0.21888801444823552}};
inline constexpr NistComponent c_G4_DIMETHYL_SULFOXIDE[] = {{6, 0.30743698710537021}, {1, 0.077400317110304442}, {8, 0.20476697797110383}, {16, 0.41039571781322154}};
inline constexpr NistComponent c_G4_ETHANE[] = {{6, 0.79887522269169908}, {1, 0.20112477730830097}};
inline constexpr NistComponent c_G4_ETHYL_ALCOHOL[] = {{6, 0.52142936609130397}, {1, 0.13127502538352509}, {8, 0.34729560852517094}};
inline constexpr NistComponent c_G4_ETHYL_CELLULOSE[] = {{1, 0.090026999999999996}, {6, 0.58518199999999998}, {8, 0.324791}};
inline constexpr NistComponent c_G4_ETHYLENE[] = {{6, 0.85628171225519822}, {1, 0.14371828774480183}};
inline constexpr NistComponent c_G4_EYE_LENS_ICRP[] = {{1, 0.096000000000000002}, {6, 0.19500000000000001}, {7, 0.057000000000000002}, {8, 0.64600000000000002}, {11, 0.001}, {15, 0.001}, {16, 0.0030000000000000001}, {17, 0.001}};
inline constexpr NistComponent c_G4_FERRIC_OXIDE[] = {{26, 0.69942604855195611}, {8, 0.30057395144804383}};
inline constexpr NistComponent c_G4_FERROBORIDE[] = {{26, 0.83780911291341298}, {5, 0.16219088708658702}};
inline constexpr NistComponent c_G4_FERROUS_OXIDE[] = {{26, 0.77730528931564569}, {8, 0.22269471068435429}};
inline constexpr NistComponent c_G4_FERROUS_SULFATE[] = {{1, 0.10825900000000001}, {7, 2.7000000000000002e-05}, {8, 0.87863600000000008}, {11, 2.2000000000000003e-05}, {16, 0.012968000000000002}, {17, 3.4000000000000007e-05}, {26, 5.4000000000000005e-05}};
inline constexpr NistComponent c_G4_FREON_12[] = {{6, 0.099335000000000007}, {9, 0.314247}, {17, 0.58641799999999999}};
inline constexpr NistComponent c_G4_FREON_12B2[] = {{6, 0.057244999999999997}, {9, 0.18109600000000001}, {35, 0.76165899999999997}};
inline constexpr NistComponent c_G4_FREON_13[] = {{6, 0.114982885017115}, {9, 0.5456214543785457}, {17, 0.3393956606043394}};
inline constexpr NistComponent c_G4_FREON_13B1[] = {{6, 0.080657986215660324}, {9, 0.38275072489425532}, {35, 0.53659128889008434}};
inline constexpr NistComponent c_G4_FREON_13I1[] = {{6, 0.061309000000000002}, {9, 0.29092400000000002}, {53, 0.64776699999999998}};
inline constexpr NistComponent c_G4_GADOLINIUM_OXYSULFIDE[] = {{64, 0.83077095448937543}, {8, 0.084525591263040989}, {16, 0.084703454247583521}};
inline constexpr NistComponent c_G4_GALLIUM_ARSENIDE[] = {{31, 0.48203003735406519}, {33, 0.51796996264593487}};
inline constexpr NistComponent c_G4_GEL_PHOTO_EMULSION[] = {{1, 0.081180000000000002}, {6, 0.41605999999999999}, {7, 0.11124000000000001}, {8, 0.38063999999999998}, {16, 0.010880000000000001}};
inline constexpr NistComponent c_G4_Pyrex_Glass[] = {{5, 0.040063919872160257}, {8, 0.53956092087815821}, {11, 0.028190943618112762}, {13, 0.011643976712046577}, {14, 0.37721924556150882}, {19, 0.0033209933580132839}};
inline constexpr NistComponent c_G4_GLASS_LEAD[] = {{8, 0.15645300000000001}, {14, 0.080865999999999993}, {22, 0.0080920000000000002}, {33, 0.0026510000000000001}, {82, 0.751938}};
inline constexpr NistComponent c_G4_GLASS_PLATE[] = {{8, 0.4598004598004598}, {11, 0.096441096441096441}, {14, 0.33655333655333658}, {20, 0.1072051072051072}};
inline constexpr NistComponent c_G4_GLUTAMINE[] = {{6, 0.41091905071989082}, {1, 0.068968636753486343}, {7, 0.1916834413010747}, {8, 0.32842887122554815}};
inline constexpr NistComponent c_G4_GLYCEROL[] = {{6, 0.39125508462763925}, {1, 0.087557649979525171}, {8, 0.52118726539283555}};
inline constexpr NistComponent c_G4_GUANINE[] = {{6, 0.39737328574255371}, {1, 0.033347558055417871}, {7, 0.46341170334403847}, {8, 0.10586745285798999}};
inline constexpr NistComponent c_G4_GYPSUM[] = {{20, 0.23277869296771098}, {16, 0.18624438028438647}, {8, 0.55755989538309292}, {1, 0.023417031364809611}};
inline constexpr NistComponent c_G4_N_HEPTANE[] = {{6, 0.83905492131026105}, {1, 0.16094507868973901}};
inline constexpr NistComponent c_G4_N_HEXANE[] = {{6, 0.83625095307179209}, {1, 0.16374904692820791}};
inline constexpr NistComponent c_G4_KAPTON[] = {{6, 0.69112781428699799}, {1, 0.026363378233987947}, {7, 0.073271320154913888}, {8, 0.20923748732410025}};
inline constexpr NistComponent c_G4_LANTHANUM_OXYBROMIDE[] = {{57, 0.59156884721343239}, {35, 0.34029297153743754}, {8, 0.068138181249130167}};
inline constexpr NistComponent c_G4_LANTHANUM_OXYSULFIDE[] = {{57, 0.81260730697763484}, {8, 0.093597869847316051}, {16, 0.093794823175049183}};
inline constexpr NistComponent c_G4_LEAD_OXIDE[] = {{8, 0.071681999999999996}, {82, 0.92831799999999998}};
inline constexpr NistComponent c_G4_LITHIUM_AMIDE[] = {{3, 0.30223092852428285}, {7, 0.60997961559118552}, {1, 0.08778945588453152}};
inline constexpr NistComponent c_G4_LITHIUM_CARBONATE[] = {{3, 0.18785030646862608}, {6, 0.16255113212542396}, {8, 0.64959856140594996}};
inline constexpr NistComponent c_G4_LITHIUM_FLUORIDE[] = {{3, 0.26755791887458868}, {9, 0.73244208112541132}};
inline constexpr NistComponent c_G4_LITHIUM_HYDRIDE[] = {{3, 0.87318268056067094}, {1, 0.12681731943932903}};
inline constexpr NistComponent c_G4_LITHIUM_IODIDE[] = {{3, 0.051851644347976726}, {53, 0.94814835565202327}};
inline constexpr NistComponent c_G4_LITHIUM_OXIDE[] = {{3, 0.46453543303563483}, {8, 0.53546456696436506}};
inline constexpr NistComponent c_G4_LITHIUM_TETRABORATE[] = {{3, 0.082072359889725222}, {5, 0.25570068677242602}, {8, 0.66222695333784876}};
inline constexpr NistComponent c_G4_LUNG_ICRP[] = {{1, 0.105}, {6, 0.083000000000000004}, {7, 0.023}, {8, 0.77900000000000003}, {11, 0.002}, {15, 0.001}, {16, 0.002}, {17, 0.0030000000000000001}, {19, 0.002}};
inline constexpr NistComponent c_G4_M3_WAX[] = {{1, 0.11431811431811431}, {6, 0.65582365582365576}, {8, 0.092183092183092175}, {12, 0.13479213479213478}, {20, 0.002883002883002883}};
inline constexpr NistComponent c_G4_MAGNESIUM_CARBONATE[] = {{12, 0.28826811501198912}, {6, 0.14245258552214662}, {8, 0.5692792994658642}};
inline constexpr NistComponent c_G4_MAGNESIUM_FLUORIDE[] = {{12, 0.39011729374996312}, {9, 0.60988270625003693}};
inline constexpr NistComponent c_G4_MAGNESIUM_OXIDE[] = {{12, 0.6030361955186937}, {8, 0.3969638044813063}};
inline constexpr NistComponent c_G4_MAGNESIUM_TETRABORATE[] = {{12, 0.13537019079769835}, {5, 0.24085388254610146}, {8, 0.62377592665620019}};
inline constexpr NistComponent c_G4_MERCURIC_IODIDE[] = {{80, 0.44145238952408433}, {53, 0.55854761047591572}};
inline constexpr NistComponent c_G4_METHANE[] = {{6, 0.74868236473128813}, {1, 0.25131763526871176}};
inline constexpr NistComponent c_G4_METHANOL[] = {{6, 0.37484481890258375}, {1, 0.12582787830609221}, {8, 0.49932730279132398}};
inline constexpr NistComponent c_G4_MIX_D_WAX[] = {{1, 0.13403999999999999}, {6, 0.77795999999999998}, {8, 0.035020000000000003}, {12, 0.038594000000000003}, {22, 0.014385999999999999}};
inline constexpr NistComponent c_G4_MS20_TISSUE[] = {{1, 0.081192}, {6, 0.58344200000000002}, {7, 0.017798000000000001}, {8, 0.18638099999999999}, {12, 0.13028699999999999}, {17, 0.00089999999999999998}};
inline constexpr NistComponent c_G4_MUSCLE_SKELETAL_ICRP[] = {{1, 0.10199999999999999}, {6, 0.14299999999999999}, {7, 0.034000000000000002}, {8, 0.70999999999999996}, {11, 0.001}, {15, 0.002}, {16, 0.0030000000000000001}, {17, 0.001}, {19, 0.0040000000000000001}};
inline constexpr NistComponent c_G4_MUSCLE_STRIATED_ICRU[] = {{1, 0.10210210210210209}, {6, 0.12312312312312312}, {7, 0.035035035035035036}, {8, 0.72972972972972971}, {11, 0.001001001001001001}, {15, 0.002002002002002002}, {16, 0.004004004004004004}, {19, 0.003003003003003003}};
inline constexpr NistComponent c_G4_MUSCLE_WITH_SUCROSE[] = {{1, 0.098234098234098233}, {6, 0.15621415621415621}, {7, 0.035451035451035458}, {8, 0.71010071010071008}};
inline constexpr NistComponent c_G4_MUSCLE_WITHOUT_SUCROSE[] = {{1, 0.101969}, {6, 0.120058}, {7, 0.035451000000000003}, {8, 0.74252200000000002}};
inline constexpr NistComponent c_G4_NAPHTHALENE[] = {{6, 0.93708769571185602}, {1, 0.062912304288143928}};
inline constexpr NistComponent c_G4_NITROBENZENE[] = {{6, 0.58536764180680034}, {1, 0.040936700493169151}, {7, 0.11377472421397986}, {8, 0.25992093348605061}};
inline constexpr NistComponent c_G4_NITROUS_OXIDE[] = {{7, 0.63648430091548769}, {8, 0.36351569908451226}};
inline constexpr NistComponent c_G4_NYLON_8062[] = {{1, 0.1035091035091035}, {6, 0.64841564841564836}, {7, 0.099536099536099515}, {8, 0.14853914853914854}};
inline constexpr NistComponent c_G4_NYLON_6_6[] = {{6, 0.63684817203505295}, {1, 0.097981190342285127}, {7, 0.12378071482704152}, {8, 0.14138992279562052}};
inline constexpr NistComponent c_G4_NYLON_6_10[] = {{1, 0.10706200000000002}, {6, 0.68044900000000008}, {7, 0.099189000000000013}, {8, 0.11330000000000001}};
inline constexpr NistComponent c_G4_NYLON_11_RILSAN[] = {{1, 0.11547588452411547}, {6, 0.72081827918172081}, {7, 0.076416923583076418}, {8, 0.087288912711087296}};
inline constexpr NistComponent c_G4_OCTANE[] = {{6, 0.84117026842029818}, {1, 0.15882973157970193}};
inline constexpr NistComponent c_G4_PARAFFIN[] = {{6, 0.85138731516944621}, {1, 0.14861268483055373}};
inline constexpr NistComponent c_G4_N_PENTANE[] = {{6, 0.83235673529724352}, {1, 0.16764326470275645}};
inline constexpr NistComponent c_G4_PHOTO_EMULSION[] = {{1, 0.0141}, {6, 0.072261000000000006}, {7, 0.01932}, {8, 0.066100999999999993}, {16, 0.00189}, {35, 0.349103}, {47, 0.474105}, {53, 0.0031199999999999999}};
inline constexpr NistComponent c_G4_PLASTIC_SC_VINYLTOLUENE[] = {{6, 0.91470853180002543}, {1, 0.085291468199974602}};
inline constexpr NistComponent c_G4_PLUTONIUM_DIOXIDE[] = {{94, 0.88408875428001943}, {8, 0.11591124571998063}};
inline constexpr NistComponent c_G4_POLYACRYLONITRILE[] = {{6, 0.679048389774075}, {1, 0.056985727055390831}, {7, 0.26396588317053415}};
inline constexpr NistComponent c_G4_POLYCARBONATE[] = {{6, 0.75574537020343691}, {1, 0.055494369081123034}, {8, 0.18876026071544011}};
inline constexpr NistComponent c_G4_POLYCHLOROSTYRENE[] = {{6, 0.693290161027735}, {1, 0.050908284183107486}, {17, 0.25580155478915761}};
inline constexpr NistComponent c_G4_POLYETHYLENE[] = {{6, 0.85628171225519822}, {1, 0.14371828774480183}};
inline constexpr NistComponent c_G4_MYLAR[] = {{6, 0.6250108323408885}, {1, 0.04196071706794325}, {8, 0.3330284505911682}};
inline constexpr NistComponent c_G4_PLEXIGLASS[] = {{6, 0.5998410708799623}, {1, 0.080541840744284249}, {8, 0.31961708837575337}};
inline constexpr NistComponent c_G4_POLYOXYMETHYLENE[] = {{6, 0.40001109243242061}, {1, 0.067137845478336802}, {8, 0.5328510620892426}};
inline constexpr NistComponent c_G4_POLYPROPYLENE[] = {{6, 0.85628171225519822}, {1, 0.14371828774480183}};
inline constexpr NistComponent c_G4_POLYSTYRENE[] = {{6, 0.9225773292943783}, {1, 0.077422670705621713}};
inline constexpr NistComponent c_G4_TEFLON[] = {{6, 0.24017852606719431}, {9, 0.75982147393280564}};
inline constexpr NistComponent c_G4_POLYTRIFLUOROCHLOROETHYLENE[] = {{6, 0.20624734470248418}, {9, 0.48935836608492056}, {17, 0.30439428921259531}};
inline constexpr NistComponent c_G4_POLYVINYL_ACETATE[] = {{6, 0.55805896873435723}, {1, 0.070248445954690111}, {8, 0.37169258531095262}};
inline constexpr NistComponent c_G4_POLYVINYL_ALCOHOL[] = {{6, 0.54529036840941658}, {1, 0.091521513247240088}, {8, 0.3631881183433433}};
inline constexpr NistComponent c_G4_POLYVINYL_BUTYRAL[] = {{6, 0.67572925775265091}, {1, 0.099237574736434395}, {8, 0.22503316751091473}};
inline constexpr NistComponent c_G4_POLYVINYL_CHLORIDE[] = {{6, 0.38435667276427204}, {1, 0.048382806238632496}, {17, 0.56726052099709534}};
inline constexpr NistComponent c_G4_POLYVINYLIDENE_CHLORIDE[] = {{6, 0.24779093272837832}, {1, 0.020794610033547243}, {17, 0.73141445723807441}};
inline constexpr NistComponent c_G4_POLYVINYLIDENE_FLUORIDE[] = {{6, 0.37513531701324954}, {1, 0.031481348172060275}, {9, 0.59338333481469019}};
inline constexpr NistComponent c_G4_POLYVINYL_PYRROLIDONE[] = {{6, 0.64839925028227141}, {1, 0.081620477839139643}, {7, 0.1260258350695867}, {8, 0.14395443680900219}};
inline constexpr NistComponent c_G4_POTASSIUM_IODIDE[] = {{19, 0.23552863286839798}, {53, 0.76447136713160202}};
inline constexpr NistComponent c_G4_POTASSIUM_OXIDE[] = {{19, 0.83014783682058291}, {8, 0.16985216317941704}};
inline constexpr NistComponent c_G4_PROPANE[] = {{6, 0.81713592047345884}, {1, 0.18286407952654118}};
inline constexpr NistComponent c_G4_lPROPANE[] = {{6, 0.81713592047345884}, {1, 0.18286407952654118}};
inline constexpr NistComponent c_G4_N_PROPYL_ALCOHOL[] = {{6, 0.59958621929556088}, {1, 0.13417936886773157}, {8, 0.2662344118367076}};
inline constexpr NistComponent c_G4_PYRIDINE[] = {{6, 0.75921067646422624}, {1, 0.063712944523619849}, {7, 0.17707637901215387}};
inline constexpr NistComponent c_G4_RUBBER_BUTYL[] = {{1, 0.14371100000000001}, {6, 0.85628899999999997}};
inline constexpr NistComponent c_G4_RUBBER_NATURAL[] = {{1, 0.118371}, {6, 0.881629}};
inline constexpr NistComponent c_G4_RUBBER_NEOPRENE[] = {{1, 0.056919999999999998}, {6, 0.54264599999999996}, {17, 0.40043400000000001}};
inline constexpr NistComponent c_G4_SILICON_DIOXIDE[] = {{14, 0.46743384179152692}, {8, 0.53256615820847297}};
inline constexpr NistComponent c_G4_SILVER_BROMIDE[] = {{47, 0.57446463222677024}, {35, 0.42553536777322981}};
inline constexpr NistComponent c_G4_SILVER_CHLORIDE[] = {{47, 0.75263482318958475}, {17, 0.24736517681041528}};
inline constexpr NistComponent c_G4_SILVER_HALIDES[] = {{35, 0.42289500000000002}, {47, 0.57374800000000004}, {53, 0.0033570000000000002}};
inline constexpr NistComponent c_G4_SILVER_IODIDE[] = {{47, 0.45945904502670715}, {53, 0.5405409549732928}};
inline constexpr NistComponent c_G4_SKIN_ICRP[] = {{1, 0.10000000000000001}, {6, 0.20399999999999999}, {7, 0.042000000000000003}, {8, 0.64500000000000002}, {11, 0.002}, {15, 0.001}, {16, 0.002}, {17, 0.0030000000000000001}, {19, 0.001}};
inline constexpr NistComponent c_G4_SODIUM_CARBONATE[] = {{11, 0.43381684519879377}, {6, 0.11332111990721931}, {8, 0.45286203489398685}};
inline constexpr NistComponent c_G4_SODIUM_IODIDE[] = {{11, 0.15337392207015901}, {53, 0.84662607792984101}};
inline constexpr NistComponent c_G4_SODIUM_MONOXIDE[] = {{11, 0.74185784077953554}, {8, 0.25814215922046446}};
inline constexpr NistComponent c_G4_SODIUM_NITRATE[] = {{11, 0.27048497292651758}, {7, 0.16479571474887064}, {8, 0.56471931232461181}};
inline constexpr NistComponent c_G4_STILBENE[] = {{6, 0.93289550955712031}, {1, 0.067104490442879694}};
inline constexpr NistComponent c_G4_SUCROSE[] = {{6, 0.42106389811984324}, {1, 0.064782068567630455}, {8, 0.5141540333125264}};
inline constexpr NistComponent c_G4_TERPHENYL[] = {{6, 0.93872818329828711}, {1, 0.061271816701712854}};
inline constexpr NistComponent c_G4_TESTIS_ICRP[] = {{1, 0.106}, {6, 0.099000000000000005}, {7, 0.02}, {8, 0.76600000000000001}, {11, 0.002}, {15, 0.001}, {16, 0.002}, {17, 0.002}, {19, 0.002}};
inline constexpr NistComponent c_G4_TETRACHLOROETHYLENE[] = {{6, 0.14485447081262098}, {17, 0.85514552918737907}};
inline constexpr NistComponent c_G4_THALLIUM_CHLORIDE[] = {{81, 0.85217962741810738}, {17, 0.14782037258189262}};
inline constexpr NistComponent c_G4_TISSUE_SOFT_ICRP[] = {{1, 0.105}, {6, 0.25600000000000001}, {7, 0.027}, {8, 0.60199999999999998}, {11, 0.001}, {15, 0.002}, {16, 0.0030000000000000001}, {17, 0.002}, {19, 0.002}};
inline constexpr NistComponent c_G4_TISSUE_SOFT_ICRU_4[] = {{1, 0.10100000000000001}, {6, 0.111}, {7, 0.025999999999999999}, {8, 0.76200000000000001}};
inline constexpr NistComponent c_G4_TISSUE_METHANE[] = {{1, 0.101869}, {6, 0.456179}, {7, 0.035172000000000002}, {8, 0.40677999999999997}};
inline constexpr NistComponent c_G4_TISSUE_PROPANE[] = {{1, 0.102672}, {6, 0.56894}, {7, 0.035021999999999998}, {8, 0.29336600000000002}};
inline constexpr NistComponent c_G4_TITANIUM_DIOXIDE[] = {{22, 0.599341623574426}, {8, 0.400658376425574}};
inline constexpr NistComponent c_G4_TOLUENE[] = {{6, 0.91248489818070722}, {1, 0.087515101819292798}};
inline constexpr NistComponent c_G4_TRICHLOROETHYLENE[] = {{6, 0.18282971917580113}, {1, 0.0076715331568876412}, {17, 0.80949874766731122}};
inline constexpr NistComponent c_G4_TRIETHYL_PHOSPHATE[] = {{6, 0.39562164814639006}, {1, 0.083001401659957119}, {8, 0.35133594940814494}, {15, 0.17004100078550796}};
inline constexpr NistComponent c_G4_TUNGSTEN_HEXAFLUORIDE[] = {{74, 0.61726612260381275}, {9, 0.38273387739618719}};
inline constexpr NistComponent c_G4_URANIUM_DICARBIDE[] = {{92, 0.90833269378477988}, {6, 0.091667306215220165}};
inline constexpr NistComponent c_G4_URANIUM_MONOCARBIDE[] = {{92, 0.9519647142692832}, {6, 0.048035285730716679}};
inline constexpr NistComponent c_G4_URANIUM_OXIDE[] = {{92, 0.88149824646952901}, {8, 0.11850175353047106}};
inline constexpr NistComponent c_G4_UREA[] = {{6, 0.19999418596544297}, {1, 0.067134032070283664}, {7, 0.46646138380421825}, {8, 0.26641039816005507}};
inline constexpr NistComponent c_G4_VALINE[] = {{6, 0.51263709038565686}, {1, 0.094645087231549963}, {7, 0.11956617909481636}, {8, 0.27315164328797692}};
inline constexpr NistComponent c_G4_VITON[] = {{1, 0.009417}, {6, 0.280555}, {9, 0.71002799999999999}};
inline constexpr NistComponent c_G4_WATER_VAPOR[] = {{1, 0.11189847784106703}, {8, 0.8881015221589329}};
inline constexpr NistComponent c_G4_XYLENE[] = {{6, 0.90505930221329822}, {1, 0.09494069778670168}};
inline constexpr NistComponent c_G4_GRAPHITE[] = {{6, 1}};
inline constexpr NistComponent c_G4_lH2[] = {{1, 1}};
inline constexpr NistComponent c_G4_lN2[] = {{7, 1}};
inline constexpr NistComponent c_G4_lO2[] = {{8, 1}};
inline constexpr NistComponent c_G4_lAr[] = {{18, 1}};
inline constexpr NistComponent c_G4_lBr[] = {{35, 1}};
inline constexpr NistComponent c_G4_lKr[] = {{36, 1}};
inline constexpr NistComponent c_G4_lXe[] = {{54, 1}};
inline constexpr NistComponent c_G4_PbWO4[] = {{8, 0.14063661945907333}, {82, 0.45536576122323241}, {74, 0.40399761931769407}};
inline constexpr NistComponent c_G4_Galactic[] = {{1, 1}};
inline constexpr NistComponent c_G4_GRAPHITE_POROUS[] = {{6, 1}};
inline constexpr NistComponent c_G4_LUCITE[] = {{1, 0.080537999999999998}, {6, 0.59984800000000005}, {8, 0.31961400000000001}};
inline constexpr NistComponent c_G4_BRASS[] = {{29, 0.57513043413413178}, {30, 0.33412189145356785}, {82, 0.090747674412300167}};
inline constexpr NistComponent c_G4_BRONZE[] = {{29, 0.84936768699675214}, {30, 0.088391491940210523}, {82, 0.06224082106303732}};
inline constexpr NistComponent c_G4_STAINLESS_STEEL[] = {{26, 0.74621287462152119}, {24, 0.16900104431152541}, {28, 0.0847860810669534}};
inline constexpr NistComponent c_G4_CR39[] = {{1, 0.066150503980657901}, {6, 0.52550460769643925}, {8, 0.40834488832290272}};
inline constexpr NistComponent c_G4_OCTADECANOL[] = {{1, 0.14159904778726398}, {6, 0.79925225717642512}, {8, 0.059148695036310879}};
inline constexpr NistComponent c_G4_KEVLAR[] = {{6, 0.70579614090574794}, {1, 0.042307426990149243}, {8, 0.13431206941687271}, {7, 0.11758436268723023}};
inline constexpr NistComponent c_G4_DACRON[] = {{6, 0.6250108323408885}, {1, 0.04196071706794325}, {8, 0.3330284505911682}};
inline constexpr NistComponent c_G4_NEOPRENE[] = {{6, 0.54264217182624397}, {1, 0.056923150025293343}, {17, 0.40043467814846267}};
inline constexpr NistComponent c_G4_CYTOSINE[] = {{1, 0.045360911976366054}, {6, 0.43242061937782778}, {7, 0.378212595643865}, {8, 0.14400587300194118}};
inline constexpr NistComponent c_G4_THYMINE[] = {{1, 0.047953926860289875}, {6, 0.47618702817420594}, {7, 0.22212931744939729}, {8, 0.25372972751610678}};
inline constexpr NistComponent c_G4_URACIL[] = {{1, 0.035969934333729629}, {6, 0.42862181903648611}, {7, 0.24992667395714627}, {8, 0.28548157267263796}};
inline constexpr NistComponent c_G4_DEOXYRIBOSE[] = {{1, 0.075146191017459688}, {6, 0.44772526950881331}, {8, 0.47712853947372691}};
inline constexpr NistComponent c_G4_PHOSPHORIC_ACID[] = {{1, 0.030856845587631469}, {15, 0.31607471680421323}, {8, 0.65306843760815525}};
inline constexpr NistComponent c_G4_DNA_DEOXYRIBOSE[] = {{1, 0.084895911859646109}, {6, 0.72259237062566195}, {8, 0.19251171751469204}};
inline constexpr NistComponent c_G4_DNA_PHOSPHATE[] = {{15, 0.32613831647592084}, {8, 0.67386168352407916}};
inline constexpr NistComponent c_G4_DNA_ADENINE[] = {{1, 0.030061022686284737}, {6, 0.44776319670015818}, {7, 0.52217578061355707}};
inline constexpr NistComponent c_G4_DNA_GUANINE[] = {{1, 0.026857170655864439}, {6, 0.40004136627987102}, {7, 0.46652318514419738}, {8, 0.10657827792006716}};
inline constexpr NistComponent c_G4_DNA_CYTOSINE[] = {{1, 0.03662096162444007}, {6, 0.4363795341047908}, {7, 0.38167522288160344}, {8, 0.1453242813891657}};
inline constexpr NistComponent c_G4_DNA_THYMINE[] = {{1, 0.040283564904424654}, {6, 0.48002353038542617}, {7, 0.223918949604727}, {8, 0.25577395510542222}};
inline constexpr NistComponent c_G4_DNA_URACIL[] = {{1, 0.027222246353737364}, {6, 0.43251116861995231}, {7, 0.25219452912009016}, {8, 0.28807205590622031}};

inline constexpr NistMaterial kNistMaterials[] = {
  {"G4_WATER", 1, 78, 1, true, 3.5801414263065614, 0.25703333929878008, 2.81743333929878, 0.091150961477294901, 3.4773000000000001, 0, 2, c_G4_WATER},
  {"G4_H", 8.3747999999999985e-05, 19.199999999999999, 3, true, 9.5835000000000008, 1.8638999999999999, 3.2717999999999998, 0.14094554810781612, 5.7272999999999996, 0, 1, c_G4_H},
  {"G4_He", 0.000166322, 41.799999999999997, 3, true, 11.1393, 2.2017000000000002, 3.6122000000000001, 0.13442958788501602, 5.8346999999999998, 0, 1, c_G4_He},
  {"G4_Li", 0.53400000000000003, 40, 1, true, 3.1221000000000001, 0.13039999999999999, 1.6396999999999999, 0.95135999999999998, 2.4992999999999999, 0.14000000000000001, 1, c_G4_Li},
  {"G4_Be", 1.8480000000000003, 63.700000000000003, 1, true, 2.7847, 0.039199999999999999, 1.6921999999999999, 0.80391999999999997, 2.4339, 0.14000000000000001, 1, c_G4_Be},
  {"G4_B", 2.3700000000000001, 76, 1, true, 2.8477000000000001, 0.030499999999999999, 1.9688000000000001, 0.56223999999999996, 2.4512, 0.14000000000000001, 1, c_G4_B},
  {"G4_C", 2, 78, 1, true, 2.9925000000000002, -0.035099999999999999, 2.4860000000000002, 0.2024, 3.0036, 0.10000000000000001, 1, c_G4_C},
  {"G4_N", 0.0011651999999999999, 82, 3, true, 10.539999999999999, 1.7378, 4.1322999999999999, 0.15350267267131468, 3.2124999999999999, 0, 1, c_G4_N},
  {"G4_O", 0.0013315099999999999, 95, 3, true, 10.7004, 1.7541, 4.3212999999999999, 0.11777577384077202, 3.2913000000000001, 0, 1, c_G4_O},
  {"G4_F", 0.00158029, 115, 3, true, 10.965299999999999, 1.8432999999999999, 4.4096000000000002, 0.11083959964151709, 3.2961999999999998, 0, 1, c_G4_F},
  {"G4_Ne", 0.00083850499999999991, 137, 3, true, 11.9041, 2.0735000000000001, 4.6421000000000001, 0.080633813872042381, 3.5771000000000002, 0, 1, c_G4_Ne},
  {"G4_Na", 0.97099999999999975, 149, 1, true, 5.0526, 0.28799999999999998, 3.1962000000000002, 0.077719999999999997, 3.6452, 0.080000000000000002, 1, c_G4_Na},
  {"G4_Mg", 1.7399999999999998, 156, 1, true, 4.5297000000000001, 0.14990000000000001, 3.0668000000000002, 0.081629999999999994, 3.6166, 0.080000000000000002, 1, c_G4_Mg},
  {"G4_Al", 2.6989999999999994, 166, 1, true, 4.2394999999999996, 0.17080000000000001, 3.0127000000000002, 0.080240000000000006, 3.6345000000000001, 0.12, 1, c_G4_Al},
  {"G4_Si", 2.3299999999999996, 173, 1, true, 4.4351000000000003, 0.2014, 2.8715000000000002, 0.14921000000000001, 3.2545999999999999, 0.14000000000000001, 1, c_G4_Si},
  {"G4_P", 2.2000000000000002, 173, 1, true, 4.5213999999999999, 0.1696, 2.7814999999999999, 0.2361, 2.9157999999999999, 0.14000000000000001, 1, c_G4_P},
  {"G4_S", 2, 180, 1, true, 4.6658999999999997, 0.158, 2.7159, 0.33992, 2.6456, 0.14000000000000001, 1, c_G4_S},
  {"G4_Cl", 0.0029947300000000001, 174, 3, true, 11.142099999999999, 1.5555000000000001, 4.2994000000000003, 0.19847510302145549, 2.9702000000000002, 0, 1, c_G4_Cl},
  {"G4_Ar", 0.0016620099999999998, 188, 3, true, 11.948, 1.7635000000000001, 4.4855, 0.19714334854189114, 2.9618000000000002, 0, 1, c_G4_Ar},
  {"G4_K", 0.86199999999999988, 190, 1, true, 5.6422999999999996, 0.3851, 3.1724000000000001, 0.19827, 2.9232999999999998, 0.10000000000000001, 1, c_G4_K},
  {"G4_Ca", 1.55, 191, 1, true, 5.0396000000000001, 0.32279999999999998, 3.1191, 0.15643000000000001, 3.0745, 0.14000000000000001, 1, c_G4_Ca},
  {"G4_Sc", 2.9889999999999994, 216, 1, true, 4.6948999999999996, 0.16400000000000001, 3.0592999999999999, 0.15754000000000001, 3.0516999999999999, 0.10000000000000001, 1, c_G4_Sc},
  {"G4_Ti", 4.5399999999999991, 233, 1, true, 4.4450000000000003, 0.095699999999999993, 3.0386000000000002, 0.15662000000000001, 3.0301999999999998, 0.12, 1, c_G4_Ti},
  {"G4_V", 6.1099999999999994, 245, 1, true, 4.2659000000000002, 0.069099999999999995, 3.0322, 0.15436, 3.0163000000000002, 0.14000000000000001, 1, c_G4_V},
  {"G4_Cr", 7.1799999999999988, 257, 1, true, 4.1780999999999997, 0.034000000000000002, 3.0451000000000001, 0.15418999999999999, 2.9895999999999998, 0.14000000000000001, 1, c_G4_Cr},
  {"G4_Mn", 7.4400000000000004, 272, 1, true, 4.2702, 0.044699999999999997, 3.1074000000000002, 0.14973, 2.9796, 0.14000000000000001, 1, c_G4_Mn},
  {"G4_Fe", 7.8739999999999988, 286, 1, true, 4.2911000000000001, -0.0011999999999999999, 3.1530999999999998, 0.14680000000000001, 2.9632000000000001, 0.12, 1, c_G4_Fe},
  {"G4_Co", 8.9000000000000004, 297, 1, true, 4.2601000000000004, -0.018700000000000001, 3.1789999999999998, 0.14474000000000001, 2.9502000000000002, 0.12, 1, c_G4_Co},
  {"G4_Ni", 8.9019999999999975, 311, 1, true, 4.3114999999999997, -0.056599999999999998, 3.1850999999999998, 0.16496, 2.843, 0.10000000000000001, 1, c_G4_Ni},
  {"G4_Cu", 8.9599999999999991, 322, 1, true, 4.4189999999999996, -0.025399999999999999, 3.2791999999999999, 0.14338999999999999, 2.9043999999999999, 0.080000000000000002, 1, c_G4_Cu},
  {"G4_Zn", 7.1329999999999991, 330, 1, true, 4.6905999999999999, 0.0048999999999999998, 3.3668, 0.14713999999999999, 2.8652000000000002, 0.080000000000000002, 1, c_G4_Zn},
  {"G4_Ga", 5.903999999999999, 334, 1, true, 4.9352999999999998, 0.22670000000000001, 3.5434000000000001, 0.094399999999999998, 3.1314000000000002, 0.14000000000000001, 1, c_G4_Ga},
  {"G4_Ge", 5.3229999999999995, 350, 1, true, 5.1410999999999998, 0.33760000000000001, 3.6095999999999999, 0.071879999999999999, 3.3306, 0.14000000000000001, 1, c_G4_Ge},
  {"G4_As", 5.7299999999999995, 347, 1, true, 5.0510000000000002, 0.1767, 3.5701999999999998, 0.065094405941347494, 3.4176000000000002, 0, 1, c_G4_As},
  {"G4_Se", 4.5, 348, 1, true, 5.3209999999999997, 0.2258, 3.6263999999999998, 0.065680000000000002, 3.4317000000000002, 0.10000000000000001, 1, c_G4_Se},
  {"G4_Br", 0.0070720999999999996, 343, 3, true, 11.730700000000001, 1.5262, 4.9898999999999996, 0.063346587836640822, 3.4670000000000001, 0, 1, c_G4_Br},
  {"G4_Kr", 0.0034783199999999996, 352, 3, true, 12.5115, 1.7158, 5.0747999999999998, 0.074455783184617597, 3.4051, 0, 1, c_G4_Kr},
  {"G4_Rb", 1.5319999999999998, 363, 1, true, 6.4775999999999998, 0.57369999999999999, 3.7995000000000001, 0.072609999999999994, 3.4177, 0.14000000000000001, 1, c_G4_Rb},
  {"G4_Sr", 2.5399999999999996, 366, 1, true, 5.9866999999999999, 0.45850000000000002, 3.6778, 0.071650000000000005, 3.4434999999999998, 0.14000000000000001, 1, c_G4_Sr},
  {"G4_Y", 4.4689999999999994, 379, 1, true, 5.4801000000000002, 0.36080000000000001, 3.5541999999999998, 0.071379999999999999, 3.4584999999999999, 0.14000000000000001, 1, c_G4_Y},
  {"G4_Zr", 6.5059999999999993, 393, 1, true, 5.1773999999999996, 0.29570000000000002, 3.4889999999999999, 0.07177, 3.4533, 0.14000000000000001, 1, c_G4_Zr},
  {"G4_Nb", 8.5700000000000003, 417, 1, true, 5.0141, 0.17849999999999999, 3.2201, 0.13883000000000001, 3.093, 0.14000000000000001, 1, c_G4_Nb},
  {"G4_Mo", 10.219999999999999, 424, 1, true, 4.8792999999999997, 0.22670000000000001, 3.2784, 0.10525, 3.2549000000000001, 0.14000000000000001, 1, c_G4_Mo},
  {"G4_Tc", 11.5, 428, 1, true, 4.7769000000000004, 0.094899999999999998, 3.1253000000000002, 0.16572000000000001, 2.9738000000000002, 0.14000000000000001, 1, c_G4_Tc},
  {"G4_Ru", 12.41, 441, 1, true, 4.7694000000000001, 0.059900000000000002, 3.0834000000000001, 0.19342000000000001, 2.8706999999999998, 0.14000000000000001, 1, c_G4_Ru},
  {"G4_Rh", 12.41, 449, 1, true, 4.8007999999999997, 0.057599999999999998, 3.1069, 0.19205, 2.8633000000000002, 0.14000000000000001, 1, c_G4_Rh},
  {"G4_Pd", 12.02, 470, 1, true, 4.9358000000000004, 0.056300000000000003, 3.0554999999999999, 0.24177999999999999, 2.7239, 0.14000000000000001, 1, c_G4_Pd},
  {"G4_Ag", 10.5, 470, 1, true, 5.0629999999999997, 0.065699999999999995, 3.1074000000000002, 0.24585000000000001, 2.6899000000000002, 0.14000000000000001, 1, c_G4_Ag},
  {"G4_Cd", 8.6500000000000004, 469, 1, true, 5.2727000000000004, 0.12809999999999999, 3.1667000000000001, 0.24609, 2.6772, 0.14000000000000001, 1, c_G4_Cd},
  {"G4_In", 7.3099999999999996, 488, 1, true, 5.5210999999999997, 0.24060000000000001, 3.2031999999999998, 0.23879, 2.7143999999999999, 0.14000000000000001, 1, c_G4_In},
  {"G4_Sn", 7.3099999999999996, 488, 1, true, 5.5339999999999998, 0.28789999999999999, 3.2959000000000001, 0.18689, 2.8576000000000001, 0.14000000000000001, 1, c_G4_Sn},
  {"G4_Sb", 6.6909999999999989, 487, 1, true, 5.6241000000000003, 0.31890000000000002, 3.3489, 0.16652, 2.9319000000000002, 0.14000000000000001, 1, c_G4_Sb},
  {"G4_Te", 6.2399999999999993, 485, 1, true, 5.7130999999999998, 0.3296, 3.4418000000000002, 0.13815, 3.0354000000000001, 0.14000000000000001, 1, c_G4_Te},
  {"G4_I", 4.9299999999999997, 491.00000000000006, 1, true, 5.9488000000000003, 0.054899999999999997, 3.2595999999999998, 0.23767445351522831, 2.7275999999999998, 0, 1, c_G4_I},
  {"G4_Xe", 0.005485359999999999, 482, 3, true, 12.7281, 1.5629999999999999, 4.7370999999999999, 0.23313077271070215, 2.7414000000000001, 0, 1, c_G4_Xe},
  {"G4_Cs", 1.8729999999999998, 488, 1, true, 6.9135, 0.54730000000000001, 3.5914000000000001, 0.18232999999999999, 2.8866000000000001, 0.14000000000000001, 1, c_G4_Cs},
  {"G4_Ba", 3.4999999999999996, 491.00000000000006, 1, true, 6.3152999999999997, 0.41899999999999998, 3.4546999999999999, 0.18268000000000001, 2.8906000000000001, 0.14000000000000001, 1, c_G4_Ba},
  {"G4_La", 6.153999999999999, 500.99999999999994, 1, true, 5.7850000000000001, 0.31609999999999999, 3.3292999999999999, 0.18590999999999999, 2.8828, 0.14000000000000001, 1, c_G4_La},
  {"G4_Ce", 6.6569999999999983, 523, 1, true, 5.7836999999999996, 0.27129999999999999, 3.3431999999999999, 0.18884999999999999, 2.8592, 0.14000000000000001, 1, c_G4_Ce},
  {"G4_Pr", 6.7099999999999991, 535, 1, true, 5.8095999999999997, 0.23330000000000001, 3.2772999999999999, 0.23265, 2.7330999999999999, 0.14000000000000001, 1, c_G4_Pr},
  {"G4_Nd", 6.8999999999999986, 546, 1, true, 5.8290000000000015, 0.19840000000000038, 3.3063000000000002, 0.23530000000000001, 2.7050000000000001, 0.14000000000000001, 1, c_G4_Nd},
  {"G4_Pm", 7.2199999999999998, 560, 1, true, 5.8224, 0.16270000000000001, 3.3199000000000001, 0.24279999999999999, 2.6674000000000002, 0.14000000000000001, 1, c_G4_Pm},
  {"G4_Sm", 7.4599999999999982, 574, 1, true, 5.8597000000000019, 0.15200000000000038, 3.3460000000000005, 0.24698000000000001, 2.6402999999999999, 0.14000000000000001, 1, c_G4_Sm},
  {"G4_Eu", 5.2429999999999994, 580, 1, true, 6.2278000000000002, 0.1888, 3.4632999999999998, 0.24448, 2.6244999999999998, 0.14000000000000001, 1, c_G4_Eu},
  {"G4_Gd", 7.9004000000000012, 591, 1, true, 5.8738000000000001, 0.10580000000000001, 3.3932000000000002, 0.25108999999999998, 2.5977000000000001, 0.14000000000000001, 1, c_G4_Gd},
  {"G4_Tb", 8.2289999999999974, 614, 1, true, 5.9044999999999996, 0.094700000000000006, 3.4224000000000001, 0.24453, 2.6055999999999999, 0.14000000000000001, 1, c_G4_Tb},
  {"G4_Dy", 8.5500000000000007, 628, 1, true, 5.9183000000000003, 0.082199999999999995, 3.4474, 0.24665000000000001, 2.5849000000000002, 0.14000000000000001, 1, c_G4_Dy},
  {"G4_Ho", 8.7949999999999982, 650, 1, true, 5.9587000000000003, 0.076100000000000001, 3.4782000000000002, 0.24637999999999999, 2.5726, 0.14000000000000001, 1, c_G4_Ho},
  {"G4_Er", 9.0660000000000007, 658, 1, true, 5.9520999999999997, 0.064799999999999996, 3.4922, 0.24823000000000001, 2.5573000000000001, 0.14000000000000001, 1, c_G4_Er},
  {"G4_Tm", 9.3209999999999997, 674, 1, true, 5.9676999999999998, 0.081199999999999994, 3.5085000000000002, 0.24889, 2.5468999999999999, 0.14000000000000001, 1, c_G4_Tm},
  {"G4_Yb", 6.7299999999999995, 684, 1, true, 6.3324999999999996, 0.11990000000000001, 3.6246, 0.25295000000000001, 2.5141, 0.14000000000000001, 1, c_G4_Yb},
  {"G4_Lu", 9.8399999999999999, 694, 1, true, 5.9785000000000004, 0.156, 3.5217999999999998, 0.24032999999999999, 2.5642999999999998, 0.14000000000000001, 1, c_G4_Lu},
  {"G4_Hf", 13.309999999999999, 705, 1, true, 5.7138999999999998, 0.19650000000000001, 3.4337, 0.22917999999999999, 2.6154999999999999, 0.14000000000000001, 1, c_G4_Hf},
  {"G4_Ta", 16.654, 718, 1, true, 5.5262000000000002, 0.2117, 3.4805000000000001, 0.17798, 2.7623000000000002, 0.14000000000000001, 1, c_G4_Ta},
  {"G4_W", 19.300000000000001, 727, 1, true, 5.4058999999999999, 0.2167, 3.496, 0.15509000000000001, 2.8447, 0.14000000000000001, 1, c_G4_W},
  {"G4_Re", 21.02, 736, 1, true, 5.3445, 0.055899999999999998, 3.4845000000000002, 0.15184, 2.8626999999999998, 0.080000000000000002, 1, c_G4_Re},
  {"G4_Os", 22.569999999999997, 746, 1, true, 5.3083, 0.089099999999999999, 3.5413999999999999, 0.12751000000000001, 2.9607999999999999, 0.10000000000000001, 1, c_G4_Os},
  {"G4_Ir", 22.419999999999998, 757, 1, true, 5.3418000000000001, 0.081900000000000001, 3.548, 0.12690000000000001, 2.9658000000000002, 0.10000000000000001, 1, c_G4_Ir},
  {"G4_Pt", 21.449999999999996, 790, 1, true, 5.4732000000000003, 0.1484, 3.6212, 0.11128, 3.0417000000000001, 0.12, 1, c_G4_Pt},
  {"G4_Au", 19.32, 790, 1, true, 5.5747, 0.2021, 3.6979000000000002, 0.097559999999999994, 3.1101000000000001, 0.14000000000000001, 1, c_G4_Au},
  {"G4_Hg", 13.545999999999999, 800, 1, true, 5.9604999999999997, 0.27560000000000001, 3.7275, 0.11014, 3.0518999999999998, 0.14000000000000001, 1, c_G4_Hg},
  {"G4_Tl", 11.720000000000001, 810, 1, true, 6.1364999999999998, 0.34910000000000002, 3.8043999999999998, 0.094549999999999995, 3.145, 0.14000000000000001, 1, c_G4_Tl},
  {"G4_Pb", 11.35, 823, 1, true, 6.2018000000000004, 0.37759999999999999, 3.8073000000000001, 0.093590000000000007, 3.1608000000000001, 0.14000000000000001, 1, c_G4_Pb},
  {"G4_Bi", 9.7469999999999981, 823, 1, true, 6.3505000000000003, 0.41520000000000001, 3.8248000000000002, 0.094100000000000003, 3.1671, 0.14000000000000001, 1, c_G4_Bi},
  {"G4_Po", 9.3199999999999985, 830, 1, true, 6.4002999999999997, 0.42670000000000002, 3.8292999999999999, 0.09282, 3.1829999999999998, 0.14000000000000001, 1, c_G4_Po},
  {"G4_At", 9.3199999999999985, 825, 1, true, 6.3811412132488945, 0.58025203551913984, 3, 0.26178472808140452, 3, 0, 1, c_G4_At},
  {"G4_Rn", 0.0090066199999999999, 794, 3, true, 13.283899999999999, 1.5367999999999999, 4.9889000000000001, 0.2079822462076456, 2.7408999999999999, 0, 1, c_G4_Rn},
  {"G4_Fr", 1, 827, 1, true, 8.655105426657979, 1.3215643690905012, 2.9999999999999996, 0.5433291193400448, 3, 0, 1, c_G4_Fr},
  {"G4_Ra", 4.9999999999999991, 826, 1, true, 7.0452000000000004, 0.59909999999999997, 3.9428000000000001, 0.088039999999999993, 3.2454000000000001, 0.14000000000000001, 1, c_G4_Ra},
  {"G4_Ac", 10.069999999999999, 841, 1, true, 6.3742000000000001, 0.45590000000000003, 3.7966000000000002, 0.085669999999999996, 3.2683, 0.14000000000000001, 1, c_G4_Ac},
  {"G4_Th", 11.720000000000001, 847, 1, true, 6.2473000000000001, 0.42020000000000002, 3.7681, 0.086550000000000002, 3.2610000000000001, 0.14000000000000001, 1, c_G4_Th},
  {"G4_Pa", 15.369999999999999, 878, 1, true, 6.0327000000000002, 0.31440000000000001, 3.5078999999999998, 0.1477, 2.9845000000000002, 0.14000000000000001, 1, c_G4_Pa},
  {"G4_U", 18.949999999999999, 890, 1, true, 5.8693999999999997, 0.22600000000000001, 3.3721000000000001, 0.19677, 2.8170999999999999, 0.14000000000000001, 1, c_G4_U},
  {"G4_Np", 20.25, 902, 1, true, 5.8148999999999997, 0.18690000000000001, 3.3690000000000002, 0.19741, 2.8081999999999998, 0.14000000000000001, 1, c_G4_Np},
  {"G4_Pu", 19.839999999999996, 921, 1, true, 5.8747999999999996, 0.15570000000000001, 3.3980999999999999, 0.20419000000000001, 2.7679, 0.14000000000000001, 1, c_G4_Pu},
  {"G4_Am", 13.67, 934, 1, true, 6.2812999999999999, 0.22739999999999999, 3.5021, 0.20308000000000001, 2.7614999999999998, 0.14000000000000001, 1, c_G4_Am},
  {"G4_Cm", 13.51, 939, 1, true, 6.3097000000000003, 0.24840000000000001, 3.516, 0.20257, 2.7578999999999998, 0.14000000000000001, 1, c_G4_Cm},
  {"G4_Bk", 13.999999999999998, 952, 1, true, 6.2911999999999999, 0.23780000000000001, 3.5186000000000002, 0.20191999999999999, 2.7559999999999998, 0.14000000000000001, 1, c_G4_Bk},
  {"G4_Cf", 9.9999999999999982, 966, 1, true, 6.6626894119377136, 0.67203674829169469, 3, 0.28279890053788176, 3, 0, 1, c_G4_Cf},
  {"G4_A-150_TISSUE", 1.1269999999999998, 65.099999999999994, 1, true, 3.1099999999999999, 0.13289999999999999, 2.6234000000000002, 0.10781957049030769, 3.4441999999999999, 0, 6, c_G4_A_150_TISSUE},
  {"G4_ACETONE", 0.78990000000000005, 64.200000000000003, 1, true, 3.4340999999999999, 0.21970000000000001, 2.6928000000000001, 0.11101097023660317, 3.4047000000000001, 0, 3, c_G4_ACETONE},
  {"G4_ACETYLENE", 0.0010966999999999999, 58.200000000000003, 3, true, 9.8419000000000008, 1.6016999999999999, 4.0073999999999996, 0.12166797399530503, 3.4277000000000002, 0, 2, c_G4_ACETYLENE},
  {"G4_ADENINE", 1.3500000000000001, 71.400000000000006, 1, true, 3.1724000000000001, 0.1295, 2.4218999999999999, 0.20908165298931627, 3.0270999999999999, 0, 3, c_G4_ADENINE},
  {"G4_ADIPOSE_TISSUE_ICRP", 0.94999999999999996, 63.20000000000001, 1, true, 3.2366999999999999, 0.1827, 2.653, 0.1027849926817279, 3.4817, 0, 7, c_G4_ADIPOSE_TISSUE_ICRP},
  {"G4_AIR", 0.0012047899999999999, 85.700000000000003, 3, true, 10.5961, 1.7418, 4.2759, 0.10914089377455813, 3.3994, 0, 4, c_G4_AIR},
  {"G4_ALANINE", 1.4199999999999999, 71.900000000000006, 1, true, 3.0964999999999998, 0.13539999999999999, 2.6335999999999999, 0.11485029629464655, 3.3525999999999998, 0, 4, c_G4_ALANINE},
  {"G4_ALUMINUM_OXIDE", 3.9699999999999998, 145.19999999999999, 1, true, 3.5681698730052931, 0.2582030800445988, 3.0582030800445987, 0.10837740282678621, 3, 0, 2, c_G4_ALUMINUM_OXIDE},
  {"G4_AMBER", 1.1000000000000001, 63.20000000000001, 1, true, 3.0701000000000001, 0.13350000000000001, 2.5609999999999999, 0.11934117845157917, 3.4098000000000002, 0, 3, c_G4_AMBER},
  {"G4_AMMONIA", 0.00082601899999999999, 53.700000000000003, 3, true, 9.8763000000000005, 1.6821999999999999, 4.1158000000000001, 0.083148620435645207, 3.6463999999999999, 0, 2, c_G4_AMMONIA},
  {"G4_ANILINE", 1.0235000000000001, 66.200000000000003, 1, true, 3.2622, 0.1618, 2.5804999999999998, 0.13134983825985871, 3.3433999999999999, 0, 3, c_G4_ANILINE},
  {"G4_ANTHRACENE", 1.2829999999999999, 69.5, 1, true, 3.1514000000000002, 0.11459999999999999, 2.5213000000000001, 0.14677735399584668, 3.2831000000000001, 0, 2, c_G4_ANTHRACENE},
  {"G4_B-100_BONE", 1.45, 85.900000000000006, 1, true, 3.4527999999999999, 0.12520000000000001, 3.0419999999999998, 0.052686617276456613, 3.7364999999999999, 0, 6, c_G4_B_100_BONE},
  {"G4_BAKELITE", 1.2499999999999998, 72.400000000000006, 1, true, 3.2582, 0.14710000000000001, 2.6055000000000001, 0.1271267132669128, 3.347, 0, 3, c_G4_BAKELITE},
  {"G4_BARIUM_FLUORIDE", 4.8899999999999997, 375.89999999999998, 1, true, 5.4122000000000003, -0.0097999999999999997, 3.3871000000000002, 0.15992032362174538, 2.8866999999999998, 0, 2, c_G4_BARIUM_FLUORIDE},
  {"G4_BARIUM_SULFATE", 4.5, 285.69999999999999, 1, true, 4.8922999999999996, -0.012800000000000001, 3.4068999999999998, 0.11747601528899117, 3.0427, 0, 3, c_G4_BARIUM_SULFATE},
  {"G4_BENZENE", 0.87864999999999993, 63.399999999999999, 1, true, 3.3269000000000002, 0.17100000000000001, 2.5091000000000001, 0.1651785899743978, 3.2174, 0, 2, c_G4_BENZENE},
  {"G4_BERYLLIUM_OXIDE", 3.0099999999999998, 93.200000000000003, 1, true, 2.9801000000000002, 0.0241, 2.5846, 0.10754545384182776, 3.4927000000000001, 0, 2, c_G4_BERYLLIUM_OXIDE},
  {"G4_BGO", 7.129999999999999, 534.10000000000002, 1, true, 5.7408999999999999, 0.045600000000000002, 3.7816000000000001, 0.095690882591344173, 3.0781000000000001, 0, 3, c_G4_BGO},
  {"G4_BLOOD_ICRP", 1.0600000000000001, 75.200000000000003, 1, true, 3.4581, 0.22389999999999999, 2.8016999999999999, 0.084918285840916763, 3.5406, 0, 10, c_G4_BLOOD_ICRP},
  {"G4_BONE_COMPACT_ICRU", 1.8499999999999999, 91.900000000000006, 1, true, 3.339, 0.094399999999999998, 3.0200999999999998, 0.058220375161520004, 3.6419000000000001, 0, 8, c_G4_BONE_COMPACT_ICRU},
  {"G4_BONE_CORTICAL_ICRP", 1.9199999999999999, 110, 1, true, 3.7153495777697452, 0.1305510571991958, 3.1063510571991957, 0.061972758142633415, 3.5918999999999999, 0, 9, c_G4_BONE_CORTICAL_ICRP},
  {"G4_BORON_CARBIDE", 2.5199999999999996, 84.700000000000003, 1, true, 2.9859, 0.0092999999999999992, 2.1006, 0.37085143655315556, 2.8075999999999999, 0, 2, c_G4_BORON_CARBIDE},
  {"G4_BORON_OXIDE", 1.8120000000000001, 99.599999999999994, 1, true, 3.6027, 0.18429999999999999, 2.7378999999999998, 0.11547265847646225, 3.3832, 0, 2, c_G4_BORON_OXIDE},
  {"G4_BRAIN_ICRP", 1.0399999999999998, 73.299999999999997, 1, true, 3.4279000000000002, 0.22059999999999999, 2.8020999999999998, 0.082552500649144184, 3.5585, 0, 9, c_G4_BRAIN_ICRP},
  {"G4_BUTANE", 0.00249343, 48.299999999999997, 3, true, 8.5632999999999999, 1.3788, 3.7524000000000002, 0.10852890678960306, 3.4883999999999999, 0, 2, c_G4_BUTANE},
  {"G4_N-BUTYL_ALCOHOL", 0.80979999999999985, 59.899999999999999, 1, true, 3.2425000000000002, 0.19370000000000001, 2.6438999999999999, 0.10081868157497156, 3.5139, 0, 3, c_G4_N_BUTYL_ALCOHOL},
  {"G4_C-552", 1.7599999999999998, 86.799999999999997, 1, true, 3.3338000000000001, 0.151, 2.7082999999999999, 0.1049200329691103, 3.4344000000000001, 0, 5, c_G4_C_552},
  {"G4_CADMIUM_TELLURIDE", 6.2000000000000002, 539.29999999999995, 1, true, 5.9096000000000002, 0.043799999999999999, 3.2835999999999999, 0.24841572379430402, 2.6665000000000001, 0, 2, c_G4_CADMIUM_TELLURIDE},
  {"G4_CADMIUM_TUNGSTATE", 7.9000000000000004, 468.30000000000001, 1, true, 5.3593999999999999, 0.0123, 3.5941000000000001, 0.1286163836489794, 2.915, 0, 3, c_G4_CADMIUM_TUNGSTATE},
  {"G4_CALCIUM_CARBONATE", 2.7999999999999998, 136.40000000000001, 1, true, 3.7738, 0.049200000000000001, 3.0548999999999999, 0.08301151103088486, 3.4119999999999999, 0, 3, c_G4_CALCIUM_CARBONATE},
  {"G4_CALCIUM_FLUORIDE", 3.1800000000000002, 166, 1, true, 4.0652999999999997, 0.067599999999999993, 3.1682999999999999, 0.069415851425311315, 3.5263, 0, 2, c_G4_CALCIUM_FLUORIDE},
  {"G4_CALCIUM_OXIDE", 3.2999999999999989, 176.09999999999999, 1, true, 4.1208999999999998, -0.0172, 3.0171000000000001, 0.12127142297252723, 3.1936, 0, 2, c_G4_CALCIUM_OXIDE},
  {"G4_CALCIUM_SULFATE", 2.96, 152.30000000000001, 1, true, 3.9388000000000001, 0.058700000000000002, 3.1229, 0.077078788100453252, 3.4495, 0, 3, c_G4_CALCIUM_SULFATE},
  {"G4_CALCIUM_TUNGSTATE", 6.0620000000000003, 395, 1, true, 5.2603, 0.032300000000000002, 3.8932000000000002, 0.062097399173008119, 3.2648999999999999, 0, 3, c_G4_CALCIUM_TUNGSTATE},
  {"G4_CARBON_DIOXIDE", 0.0018421199999999998, 85, 3, true, 10.153700000000001, 1.6294, 4.1825000000000001, 0.11767588149668126, 3.3227000000000002, 0, 2, c_G4_CARBON_DIOXIDE},
  {"G4_CARBON_TETRACHLORIDE", 1.5939999999999999, 166.30000000000001, 1, true, 4.7712000000000003, 0.17730000000000001, 2.9165000000000001, 0.19018061380256593, 3.0116000000000001, 0, 2, c_G4_CARBON_TETRACHLORIDE},
  {"G4_CELLULOSE_CELLOPHANE", 1.4199999999999999, 77.599999999999994, 1, true, 3.2646999999999999, 0.158, 2.6778, 0.11151056239356513, 3.3809999999999998, 0, 3, c_G4_CELLULOSE_CELLOPHANE},
  {"G4_CELLULOSE_BUTYRATE", 1.1999999999999997, 74.599999999999994, 1, true, 3.3496999999999999, 0.1794, 2.6808999999999998, 0.11443530346368179, 3.3738000000000001, 0, 3, c_G4_CELLULOSE_BUTYRATE},
  {"G4_CELLULOSE_NITRATE", 1.4899999999999998, 87, 1, true, 3.4762, 0.18970000000000001, 2.7252999999999998, 0.11813106756016416, 3.3237000000000001, 0, 4, c_G4_CELLULOSE_NITRATE},
  {"G4_CERIC_SULFATE", 1.0299999999999998, 76.700000000000003, 1, true, 3.5211999999999999, 0.23630000000000001, 2.8769, 0.076662907736881725, 3.5607000000000002, 0, 5, c_G4_CERIC_SULFATE},
  {"G4_CESIUM_FLUORIDE", 4.1149999999999993, 440.69999999999999, 1, true, 5.9046000000000003, 0.0083999999999999995, 3.3374000000000001, 0.22052805576678666, 2.7280000000000002, 0, 2, c_G4_CESIUM_FLUORIDE},
  {"G4_CESIUM_IODIDE", 4.5099999999999998, 553.10000000000002, 1, true, 6.2807000000000004, 0.0395, 3.3353000000000002, 0.25381416436597765, 2.6657000000000002, 0, 2, c_G4_CESIUM_IODIDE},
  {"G4_CHLOROBENZENE", 1.1057999999999997, 89.099999999999994, 1, true, 3.8201000000000001, 0.1714, 2.9272, 0.098548130758717142, 3.3797000000000001, 0, 3, c_G4_CHLOROBENZENE},
  {"G4_CHLOROFORM", 1.4832000000000001, 156, 1, true, 4.7054999999999998, 0.17860000000000001, 2.9581, 0.16960259850102152, 3.0627, 0, 3, c_G4_CHLOROFORM},
  {"G4_CONCRETE", 2.2999999999999998, 135.19999999999999, 1, true, 3.9464000000000001, 0.13009999999999999, 3.0466000000000002, 0.07515613109494472, 3.5467, 0, 10, c_G4_CONCRETE},
  {"G4_CYCLOHEXANE", 0.77899999999999991, 56.399999999999999, 1, true, 3.1543999999999999, 0.17280000000000001, 2.5548999999999999, 0.12036930138666402, 3.4278, 0, 2, c_G4_CYCLOHEXANE},
  {"G4_1,2-DICHLOROBENZENE", 1.3048, 106.5, 1, true, 4.0347999999999997, 0.15870000000000001, 2.8275999999999999, 0.16010187884119875, 3.0836000000000001, 0, 3, c_G4_1_2_DICHLOROBENZENE},
  {"G4_DICHLORODIETHYL_ETHER", 1.2199, 103.3, 1, true, 4.0134999999999996, 0.17730000000000001, 3.1585999999999999, 0.067992821860826697, 3.5249999999999999, 0, 4, c_G4_DICHLORODIETHYL_ETHER},
  {"G4_1,2-DICHLOROETHANE", 1.2350999999999999, 111.90000000000001, 1, true, 4.1848999999999998, 0.13750000000000001, 2.9529000000000001, 0.13381865353473649, 3.1675, 0, 3, c_G4_1_2_DICHLOROETHANE},
  {"G4_DIETHYL_ETHER", 0.71377999999999986, 60, 1, true, 3.3721000000000001, 0.22309999999999999, 2.6745000000000001, 0.1055010785985448, 3.4586000000000001, 0, 3, c_G4_DIETHYL_ETHER},
  {"G4_N,N-DIMETHYL_FORMAMIDE", 0.94869999999999999, 66.599999999999994, 1, true, 3.3311000000000002, 0.19769999999999999, 2.6686000000000001, 0.11471458353946101, 3.371, 0, 4, c_G4_N_N_DIMETHYL_FORMAMIDE},
  {"G4_DIMETHYL_SULFOXIDE", 1.1013999999999999, 98.599999999999994, 1, true, 3.9843999999999999, 0.2021, 3.1263000000000001, 0.066192322195771203, 3.5708000000000002, 0, 4, c_G4_DIMETHYL_SULFOXIDE},
  {"G4_ETHANE", 0.0012532400000000001, 45.399999999999999, 3, true, 9.1043000000000003, 1.5106999999999999, 3.8742999999999999, 0.096265628287385122, 3.6095000000000002, 0, 2, c_G4_ETHANE},
  {"G4_ETHYL_ALCOHOL", 0.7893, 62.899999999999999, 1, true, 3.3698999999999999, 0.2218, 2.7052, 0.098782502552466636, 3.4834000000000001, 0, 3, c_G4_ETHYL_ALCOHOL},
  {"G4_ETHYL_CELLULOSE", 1.1299999999999997, 69.299999999999997, 1, true, 3.2414999999999998, 0.16830000000000001, 2.6526999999999998, 0.11077606905888975, 3.4098000000000002, 0, 3, c_G4_ETHYL_CELLULOSE},
  {"G4_ETHYLENE", 0.0011749699999999998, 50.700000000000003, 3, true, 9.4380000000000006, 1.5528, 3.9327000000000001, 0.1063543556950045, 3.5387, 0, 2, c_G4_ETHYLENE},
  {"G4_EYE_LENS_ICRP", 1.0700000000000001, 73.299999999999997, 1, true, 3.3719999999999999, 0.20699999999999999, 2.7446000000000002, 0.096895880374259086, 3.4550000000000001, 0, 8, c_G4_EYE_LENS_ICRP},
  {"G4_FERRIC_OXIDE", 5.2000000000000002, 227.30000000000001, 1, true, 4.2244999999999999, -0.0074000000000000003, 3.2572999999999999, 0.10477729262424439, 3.1313, 0, 2, c_G4_FERRIC_OXIDE},
  {"G4_FERROBORIDE", 7.1500000000000004, 261, 1, true, 4.2057000000000002, -0.098799999999999999, 3.1749000000000001, 0.12911381835179805, 3.024, 0, 2, c_G4_FERROBORIDE},
  {"G4_FERROUS_OXIDE", 5.7000000000000002, 248.59999999999999, 1, true, 4.3174999999999999, -0.027900000000000001, 3.2002000000000002, 0.12959154600776848, 3.0167999999999999, 0, 2, c_G4_FERROUS_OXIDE},
  {"G4_FERROUS_SULFATE", 1.024, 76.400000000000006, 1, true, 3.5183, 0.23780000000000001, 2.8254000000000001, 0.087584422833853301, 3.4923000000000002, 0, 7, c_G4_FERROUS_SULFATE},
  {"G4_FREON-12", 1.1199999999999999, 143, 1, true, 4.8250999999999999, 0.30349999999999999, 3.2658999999999998, 0.079772872198122916, 3.4626000000000001, 0, 3, c_G4_FREON_12},
  {"G4_FREON-12B2", 1.8, 284.89999999999998, 1, true, 5.7976000000000001, 0.34060000000000001, 3.7955999999999999, 0.051434592757098191, 3.5565000000000002, 0, 3, c_G4_FREON_12B2},
  {"G4_FREON-13", 0.94999999999999996, 126.59999999999999, 1, true, 4.7483000000000004, 0.3659, 3.2336999999999998, 0.072369083550471766, 3.5550999999999999, 0, 3, c_G4_FREON_13},
  {"G4_FREON-13B1", 1.5, 210.5, 1, true, 5.3555000000000001, 0.35220000000000001, 3.7553999999999998, 0.039248291852256749, 3.7193999999999998, 0, 3, c_G4_FREON_13B1},
  {"G4_FREON-13I1", 1.8, 293.5, 1, true, 5.8773999999999997, 0.28470000000000001, 3.7280000000000002, 0.091119260440050656, 3.1657999999999999, 0, 3, c_G4_FREON_13I1},
  {"G4_GADOLINIUM_OXYSULFIDE", 7.4400000000000004, 493.30000000000001, 1, true, 5.5347, -0.1774, 3.4045000000000001, 0.22159944002854165, 2.6299999999999999, 0, 3, c_G4_GADOLINIUM_OXYSULFIDE},
  {"G4_GALLIUM_ARSENIDE", 5.3099999999999996, 384.89999999999998, 1, true, 5.3299000000000003, 0.1764, 3.6419999999999999, 0.071518405963764298, 3.3355999999999999, 0, 2, c_G4_GALLIUM_ARSENIDE},
  {"G4_GEL_PHOTO_EMULSION", 1.2913999999999999, 74.799999999999997, 1, true, 3.2686999999999999, 0.1709, 2.7058, 0.10101660391971261, 3.4418000000000002, 0, 5, c_G4_GEL_PHOTO_EMULSION},
  {"G4_Pyrex_Glass", 2.23, 134, 1, true, 3.9708000000000001, 0.1479, 2.9933000000000001, 0.082695375315790423, 3.5224000000000002, 0, 6, c_G4_Pyrex_Glass},
  {"G4_GLASS_LEAD", 6.2199999999999998, 526.39999999999998, 1, true, 5.8475999999999999, 0.061400000000000003, 3.8146, 0.095442549477206484, 3.0739999999999998, 0, 5, c_G4_GLASS_LEAD},
  {"G4_GLASS_PLATE", 2.3999999999999995, 145.40000000000001, 1, true, 4.0602, 0.1237, 3.0649000000000002, 0.076772545193348765, 3.5381, 0, 4, c_G4_GLASS_PLATE},
  {"G4_GLUTAMINE", 1.4599999999999997, 73.299999999999997, 1, true, 3.1166999999999998, 0.13469999999999999, 2.6301000000000001, 0.11930575483878396, 3.3254000000000001, 0, 4, c_G4_GLUTAMINE},
  {"G4_GLYCEROL", 1.2612999999999999, 72.599999999999994, 1, true, 3.2267000000000001, 0.1653, 2.6861999999999999, 0.1016939178298016, 3.4481000000000002, 0, 3, c_G4_GLYCEROL},
  {"G4_GUANINE", 1.5800000000000001, 75, 1, true, 3.1171000000000002, 0.1163, 2.4296000000000002, 0.20530744040714968, 3.0186000000000002, 0, 4, c_G4_GUANINE},
  {"G4_GYPSUM", 2.3199999999999998, 129.69999999999999, 1, true, 3.8382000000000001, 0.099500000000000005, 3.1206, 0.069486874913997645, 3.5133999999999999, 0, 4, c_G4_GYPSUM},
  {"G4_N-HEPTANE", 0.68376000000000003, 54.399999999999999, 1, true, 3.1978, 0.1928, 2.5706000000000002, 0.11254078113869398, 3.4885000000000002, 0, 2, c_G4_N_HEPTANE},
  {"G4_N-HEXANE", 0.66029999999999989, 54, 1, true, 3.2155999999999998, 0.19839999999999999, 2.5756999999999999, 0.11086220669130747, 3.5026999999999999, 0, 2, c_G4_N_HEXANE},
  {"G4_KAPTON", 1.4199999999999999, 79.599999999999994, 1, true, 3.3496999999999999, 0.15090000000000001, 2.5630999999999999, 0.15970811018620801, 3.1920999999999999, 0, 4, c_G4_KAPTON},
  {"G4_LANTHANUM_OXYBROMIDE", 6.2800000000000002, 439.69999999999999, 1, true, 5.4665999999999997, -0.035000000000000003, 3.3288000000000002, 0.17829393968578797, 2.8456999999999999, 0, 3, c_G4_LANTHANUM_OXYBROMIDE},
  {"G4_LANTHANUM_OXYSULFIDE", 5.8600000000000003, 421.19999999999999, 1, true, 5.4554529408190664, -0.12806691842718065, 3.2394330815728192, 0.22580158162181396, 2.7075, 0, 3, c_G4_LANTHANUM_OXYSULFIDE},
  {"G4_LEAD_OXIDE", 9.5299999999999994, 766.70000000000005, 1, true, 6.2161999999999997, 0.0356, 3.5455999999999999, 0.19646418374601746, 2.7299000000000002, 0, 2, c_G4_LEAD_OXIDE},
  {"G4_LITHIUM_AMIDE", 1.1779999999999999, 55.5, 1, true, 2.7961, 0.019800000000000002, 2.5152000000000001, 0.087403359877894868, 3.7534000000000001, 0, 3, c_G4_LITHIUM_AMIDE},
  {"G4_LITHIUM_CARBONATE", 2.1099999999999999, 87.900000000000006, 1, true, 3.2029000000000001, 0.055100000000000003, 2.6598000000000002, 0.099359302703422109, 3.5417000000000001, 0, 3, c_G4_LITHIUM_CARBONATE},
  {"G4_LITHIUM_FLUORIDE", 2.6349999999999993, 94, 1, true, 3.1667000000000001, 0.017100000000000001, 2.7048999999999999, 0.075923565570935436, 3.7477999999999998, 0, 2, c_G4_LITHIUM_FLUORIDE},
  {"G4_LITHIUM_HYDRIDE", 0.81999999999999995, 36.5, 1, true, 2.3580000000000001, -0.098799999999999999, 1.4515, 0.90565470880065768, 2.5849000000000002, 0, 2, c_G4_LITHIUM_HYDRIDE},
  {"G4_LITHIUM_IODIDE", 3.4940000000000002, 485.10000000000002, 1, true, 6.2671000000000001, 0.089200000000000002, 3.3702000000000001, 0.23274169817805315, 2.7145999999999999, 0, 2, c_G4_LITHIUM_IODIDE},
  {"G4_LITHIUM_OXIDE", 2.0129999999999999, 73.599999999999994, 1, true, 2.9340000000000002, -0.0511, 2.5874000000000001, 0.080343407798566538, 3.7877999999999998, 0, 2, c_G4_LITHIUM_OXIDE},
  {"G4_LITHIUM_TETRABORATE", 2.4399999999999999, 94.599999999999994, 1, true, 3.2092999999999998, 0.073700000000000002, 2.6501999999999999, 0.11075798207820396, 3.4388999999999998, 0, 3, c_G4_LITHIUM_TETRABORATE},
  {"G4_LUNG_ICRP", 1.0399999999999998, 75.299999999999997, 1, true, 3.4708000000000001, 0.2261, 2.8001, 0.085882606831471406, 3.5352999999999999, 0, 9, c_G4_LUNG_ICRP},
  {"G4_M3_WAX", 1.05, 67.900000000000006, 1, true, 3.254, 0.15229999999999999, 2.7528999999999999, 0.078636190828243899, 3.6412, 0, 5, c_G4_M3_WAX},
  {"G4_MAGNESIUM_CARBONATE", 2.9579999999999997, 118, 1, true, 3.4319000000000002, 0.085999999999999993, 2.7997000000000001, 0.092190163878935263, 3.5003000000000002, 0, 3, c_G4_MAGNESIUM_CARBONATE},
  {"G4_MAGNESIUM_FLUORIDE", 3, 134.30000000000001, 1, true, 3.7105000000000001, 0.13689999999999999, 2.863, 0.079338481248603474, 3.6484999999999999, 0, 2, c_G4_MAGNESIUM_FLUORIDE},
  {"G4_MAGNESIUM_OXIDE", 3.5800000000000001, 143.80000000000001, 1, true, 3.6404000000000001, 0.057500000000000002, 2.8580000000000001, 0.08312552554383619, 3.5968, 0, 2, c_G4_MAGNESIUM_OXIDE},
  {"G4_MAGNESIUM_TETRABORATE", 2.5299999999999994, 108.3, 1, true, 3.4327999999999999, 0.1147, 2.7635000000000001, 0.097037349413615262, 3.4893000000000001, 0, 3, c_G4_MAGNESIUM_TETRABORATE},
  {"G4_MERCURIC_IODIDE", 6.3600000000000003, 684.5, 1, true, 6.3787000000000003, 0.104, 3.4727999999999999, 0.21514242112032375, 2.7263999999999999, 0, 2, c_G4_MERCURIC_IODIDE},
  {"G4_METHANE", 0.00066715100000000005, 41.700000000000003, 3, true, 9.5243000000000002, 1.6263000000000001, 3.9716, 0.092537385006173936, 3.6257000000000001, 0, 2, c_G4_METHANE},
  {"G4_METHANOL", 0.79139999999999988, 67.599999999999994, 1, true, 3.516, 0.25290000000000001, 2.7639, 0.089697750286549749, 3.5476999999999999, 0, 3, c_G4_METHANOL},
  {"G4_MIX_D_WAX", 0.98999999999999977, 60.899999999999999, 1, true, 3.0779999999999998, 0.1371, 2.7145000000000001, 0.074898595297176837, 3.6823000000000001, 0, 5, c_G4_MIX_D_WAX},
  {"G4_MS20_TISSUE", 1, 75.099999999999994, 1, true, 3.5341, 0.19969999999999999, 2.8033000000000001, 0.082942587244433569, 3.6061000000000001, 0, 6, c_G4_MS20_TISSUE},
  {"G4_MUSCLE_SKELETAL_ICRP", 1.05, 75.299999999999997, 1, true, 3.470140545494774, 0.2372586363587435, 2.0372586363587435, 0.40766874947998943, 3, 0, 9, c_G4_MUSCLE_SKELETAL_ICRP},
  {"G4_MUSCLE_STRIATED_ICRU", 1.0399999999999998, 74.700000000000003, 1, true, 3.4636, 0.22489999999999999, 2.8031999999999999, 0.085076410391389759, 3.5383, 0, 8, c_G4_MUSCLE_STRIATED_ICRU},
  {"G4_MUSCLE_WITH_SUCROSE", 1.1100000000000001, 74.299999999999997, 1, true, 3.391, 0.20979999999999999, 2.7549999999999999, 0.09481297491102629, 3.4699, 0, 4, c_G4_MUSCLE_WITH_SUCROSE},
  {"G4_MUSCLE_WITHOUT_SUCROSE", 1.0700000000000001, 74.200000000000003, 1, true, 3.4216000000000002, 0.21870000000000001, 2.7679999999999998, 0.091427392417608111, 3.4982000000000002, 0, 4, c_G4_MUSCLE_WITHOUT_SUCROSE},
  {"G4_NAPHTHALENE", 1.1449999999999998, 68.400000000000006, 1, true, 3.227340459028849, 0.21458424891045241, 2.0145842489104524, 0.38394092482232262, 3, 0, 2, c_G4_NAPHTHALENE},
  {"G4_NITROBENZENE", 1.1986699999999999, 75.799999999999997, 1, true, 3.4073000000000002, 0.1777, 2.6629999999999998, 0.12728507971609959, 3.3090999999999999, 0, 4, c_G4_NITROBENZENE},
  {"G4_NITROUS_OXIDE", 0.0018309400000000001, 84.900000000000006, 3, true, 10.157500000000001, 1.6476999999999999, 4.1565000000000003, 0.11992728196560616, 3.3317999999999999, 0, 2, c_G4_NITROUS_OXIDE},
  {"G4_NYLON-8062", 1.0800000000000001, 64.299999999999997, 1, true, 3.125, 0.15029999999999999, 2.6004, 0.11512687208794926, 3.4043999999999999, 0, 4, c_G4_NYLON_8062},
  {"G4_NYLON-6-6", 1.1399999999999999, 63.899999999999999, 1, true, 3.0634000000000001, 0.1336, 2.5834000000000001, 0.11818561597422197, 3.3826000000000001, 0, 4, c_G4_NYLON_6_6},
  {"G4_NYLON-6-10", 1.1399999999999999, 63.20000000000001, 1, true, 3.0333000000000001, 0.13039999999999999, 2.5680999999999998, 0.11851584707759873, 3.3912, 0, 4, c_G4_NYLON_6_10},
  {"G4_NYLON-11_RILSAN", 1.425, 61.599999999999994, 1, true, 2.7513999999999998, 0.067799999999999999, 2.4281000000000001, 0.14868391995292271, 3.2576000000000001, 0, 4, c_G4_NYLON_11_RILSAN},
  {"G4_OCTANE", 0.7026, 54.700000000000003, 1, true, 3.1833999999999998, 0.18820000000000001, 2.5663999999999998, 0.113875358533193, 3.4775999999999998, 0, 2, c_G4_OCTANE},
  {"G4_PARAFFIN", 0.93000000000000005, 55.899999999999999, 1, true, 2.9550999999999998, 0.12889999999999999, 2.5084, 0.12086277645706756, 3.4287999999999998, 0, 2, c_G4_PARAFFIN},
  {"G4_N-PENTANE", 0.62619999999999987, 53.600000000000001, 1, true, 3.2504, 0.20860000000000001, 2.5855000000000001, 0.10809056136239793, 3.5265, 0, 2, c_G4_N_PENTANE},
  {"G4_PHOTO_EMULSION", 3.8149999999999999, 331, 1, true, 5.3319000000000001, 0.1009, 3.4866000000000001, 0.12398195954046741, 3.0093999999999999, 0, 8, c_G4_PHOTO_EMULSION},
  {"G4_PLASTIC_SC_VINYLTOLUENE", 1.032, 64.700000000000003, 1, true, 3.1997, 0.1464, 2.4855, 0.16102309181468938, 3.2393000000000001, 0, 2, c_G4_PLASTIC_SC_VINYLTOLUENE},
  {"G4_PLUTONIUM_DIOXIDE", 11.459999999999999, 746.5, 1, true, 5.9718999999999998, -0.2311, 3.5554000000000001, 0.20593618680470399, 2.6522000000000001, 0, 2, c_G4_PLUTONIUM_DIOXIDE},
  {"G4_POLYACRYLONITRILE", 1.1699999999999999, 69.599999999999994, 1, true, 3.2458999999999998, 0.15040000000000001, 2.5158999999999998, 0.16273471664230815, 3.1974999999999998, 0, 3, c_G4_POLYACRYLONITRILE},
  {"G4_POLYCARBONATE", 1.1999999999999997, 73.099999999999994, 1, true, 3.3201000000000001, 0.16059999999999999, 2.6225000000000001, 0.12860106206922992, 3.3288000000000002, 0, 3, c_G4_POLYCARBONATE},
  {"G4_POLYCHLOROSTYRENE", 1.3, 81.700000000000003, 1, true, 3.4659, 0.12379999999999999, 2.9241000000000001, 0.075305703741919902, 3.5440999999999998, 0, 3, c_G4_POLYCHLOROSTYRENE},
  {"G4_POLYETHYLENE", 0.93999999999999984, 57.399999999999999, 1, true, 3.0015999999999998, 0.13700000000000001, 2.5177, 0.12108195244255707, 3.4291999999999998, 0, 2, c_G4_POLYETHYLENE},
  {"G4_MYLAR", 1.3999999999999999, 78.700000000000003, 1, true, 3.3262, 0.15620000000000001, 2.6507000000000001, 0.12678186173203754, 3.3075999999999999, 0, 3, c_G4_MYLAR},
  {"G4_PLEXIGLASS", 1.1899999999999999, 74, 1, true, 3.3296649357201598, 0.24195717167608066, 2.0419571716760805, 0.3798715676322823, 3, 0, 3, c_G4_PLEXIGLASS},
  {"G4_POLYOXYMETHYLENE", 1.425, 77.400000000000006, 1, true, 3.2514196762538834, 0.22993102844159541, 2.0299310284415952, 0.37595133045971674, 3, 0, 3, c_G4_POLYOXYMETHYLENE},
  {"G4_POLYPROPYLENE", 0.90000000000000002, 56.5, 1, true, 3.0318381925248734, 0.1331267410965194, 2.4619267410965198, 0.15044619122895714, 3.2854999999999999, 0, 2, c_G4_POLYPROPYLENE},
  {"G4_POLYSTYRENE", 1.0600000000000001, 68.700000000000003, 1, true, 3.2999000000000001, 0.16470000000000001, 2.5030999999999999, 0.16454092790141286, 3.2223999999999999, 0, 2, c_G4_POLYSTYRENE},
  {"G4_TEFLON", 2.2000000000000002, 99.099999999999994, 1, true, 3.4161000000000001, 0.1648, 2.7404000000000002, 0.10605773536806372, 3.4045999999999998, 0, 2, c_G4_TEFLON},
  {"G4_POLYTRIFLUOROCHLOROETHYLENE", 2.1000000000000001, 120.7, 1, true, 3.8551000000000002, 0.1714, 3.0265, 0.07726675925770507, 3.5085000000000002, 0, 3, c_G4_POLYTRIFLUOROCHLOROETHYLENE},
  {"G4_POLYVINYL_ACETATE", 1.1899999999999999, 73.700000000000003, 1, true, 3.3309000000000002, 0.1769, 2.6747000000000001, 0.11442444561549762, 3.3761999999999999, 0, 3, c_G4_POLYVINYL_ACETATE},
  {"G4_POLYVINYL_ALCOHOL", 1.3, 69.700000000000003, 1, true, 3.1115102112494544, 0.22406073687256189, 2.024060736872562, 0.35659677399749817, 3, 0, 3, c_G4_POLYVINYL_ALCOHOL},
  {"G4_POLYVINYL_BUTYRAL", 1.1199999999999999, 67.200000000000003, 1, true, 3.1865000000000001, 0.1555, 2.6185999999999998, 0.11544979096405726, 3.3982999999999999, 0, 3, c_G4_POLYVINYL_BUTYRAL},
  {"G4_POLYVINYL_CHLORIDE", 1.3, 108.2, 1, true, 4.0532000000000004, 0.15590000000000001, 2.9415, 0.12438320478921007, 3.2103999999999999, 0, 3, c_G4_POLYVINYL_CHLORIDE},
  {"G4_POLYVINYLIDENE_CHLORIDE", 1.6999999999999997, 134.30000000000001, 1, true, 4.2506000000000004, 0.13139999999999999, 2.9009, 0.15467700267838647, 3.1019999999999999, 0, 3, c_G4_POLYVINYLIDENE_CHLORIDE},
  {"G4_POLYVINYLIDENE_FLUORIDE", 1.7599999999999998, 88.799999999999997, 1, true, 3.3793000000000002, 0.17169999999999999, 2.7374999999999998, 0.10316245596694953, 3.4199999999999999, 0, 3, c_G4_POLYVINYLIDENE_FLUORIDE},
  {"G4_POLYVINYL_PYRROLIDONE", 1.2499999999999998, 67.700000000000003, 1, true, 3.1017000000000001, 0.13239999999999999, 2.5867, 0.12504631122055093, 3.3325999999999998, 0, 4, c_G4_POLYVINYL_PYRROLIDONE},
  {"G4_POTASSIUM_IODIDE", 3.1299999999999994, 431.89999999999998, 1, true, 6.1087999999999996, 0.10440000000000001, 3.3441999999999998, 0.22053097193235657, 2.7557999999999998, 0, 2, c_G4_POTASSIUM_IODIDE},
  {"G4_POTASSIUM_OXIDE", 2.3199999999999998, 189.90000000000001, 1, true, 4.6463000000000001, 0.048000000000000001, 3.0110000000000001, 0.16789396401816845, 3.0121000000000002, 0, 2, c_G4_POTASSIUM_OXIDE},
  {"G4_PROPANE", 0.0018793900000000001, 47.100000000000001, 3, true, 8.7878000000000007, 1.4326000000000001, 3.7997999999999998, 0.099146654304074988, 3.5920000000000001, 0, 2, c_G4_PROPANE},
  {"G4_lPROPANE", 0.43000000000000005, 52, 1, true, 3.5529000000000002, 0.28610000000000002, 2.6568000000000001, 0.10328498350630468, 3.5619999999999998, 0, 2, c_G4_lPROPANE},
  {"G4_N-PROPYL_ALCOHOL", 0.8035000000000001, 61.099999999999994, 1, true, 3.291512736237582, 0.21704410670232371, 2.0170441067023237, 0.39300200394506557, 3, 0, 3, c_G4_N_PROPYL_ALCOHOL},
  {"G4_PYRIDINE", 0.98189999999999988, 66.200000000000003, 1, true, 3.3148, 0.16700000000000001, 2.5245000000000002, 0.16399275325865792, 3.1977000000000002, 0, 3, c_G4_PYRIDINE},
  {"G4_RUBBER_BUTYL", 0.91999999999999982, 56.5, 1, true, 2.9914999999999998, 0.13469999999999999, 2.5154000000000001, 0.12106506380480049, 3.4296000000000002, 0, 2, c_G4_RUBBER_BUTYL},
  {"G4_RUBBER_NATURAL", 0.91999999999999982, 59.799999999999997, 1, true, 3.1272000000000002, 0.1512, 2.4815, 0.15057518704138431, 3.2879, 0, 2, c_G4_RUBBER_NATURAL},
  {"G4_RUBBER_NEOPRENE", 1.23, 93, 1, true, 3.7911000000000001, 0.15010000000000001, 2.9460999999999999, 0.097622254683696189, 3.3632, 0, 3, c_G4_RUBBER_NEOPRENE},
  {"G4_SILICON_DIOXIDE", 2.3199999999999998, 139.19999999999999, 1, true, 4.0029000000000003, 0.13850000000000001, 3.0024999999999999, 0.084074777279540089, 3.5064000000000002, 0, 2, c_G4_SILICON_DIOXIDE},
  {"G4_SILVER_BROMIDE", 6.4729999999999981, 486.60000000000002, 1, true, 5.6139000000000001, 0.035200000000000002, 3.2109000000000001, 0.24581382276664832, 2.6819999999999999, 0, 2, c_G4_SILVER_BROMIDE},
  {"G4_SILVER_CHLORIDE", 5.5599999999999987, 398.39999999999998, 1, true, 5.3437000000000001, -0.013899999999999999, 3.2021999999999999, 0.2296908650392781, 2.7040999999999999, 0, 2, c_G4_SILVER_CHLORIDE},
  {"G4_SILVER_HALIDES", 6.4699999999999998, 487.10000000000002, 1, true, 5.6166, 0.035299999999999998, 3.2117, 0.24593996546060487, 2.6814, 0, 3, c_G4_SILVER_HALIDES},
  {"G4_SILVER_IODIDE", 6.0099999999999998, 543.5, 1, true, 5.9341999999999997, 0.014800000000000001, 3.2907999999999999, 0.25059562526279278, 2.6572, 0, 2, c_G4_SILVER_IODIDE},
  {"G4_SKIN_ICRP", 1.0900000000000001, 72.700000000000003, 1, true, 3.3546, 0.2019, 2.7526000000000002, 0.094599472859174236, 3.4643000000000002, 0, 9, c_G4_SKIN_ICRP},
  {"G4_SODIUM_CARBONATE", 2.532, 125.00000000000001, 1, true, 3.7178, 0.12870000000000001, 2.8591000000000002, 0.087145054856152426, 3.5638000000000001, 0, 3, c_G4_SODIUM_CARBONATE},
  {"G4_SODIUM_IODIDE", 3.6669999999999994, 452, 1, true, 6.0571999999999999, 0.1203, 3.5920000000000001, 0.12516272020637612, 3.0398000000000001, 0, 2, c_G4_SODIUM_IODIDE},
  {"G4_SODIUM_MONOXIDE", 2.2699999999999996, 148.80000000000001, 1, true, 4.1891999999999996, 0.16520000000000001, 2.9792999999999998, 0.075006647370109383, 3.6943000000000001, 0, 2, c_G4_SODIUM_MONOXIDE},
  {"G4_SODIUM_NITRATE", 2.2610000000000001, 114.59999999999999, 1, true, 3.6501999999999999, 0.15340000000000001, 2.8220999999999998, 0.093911583702868187, 3.5097, 0, 3, c_G4_SODIUM_NITRATE},
  {"G4_STILBENE", 0.97070000000000001, 67.700000000000003, 1, true, 3.3679999999999999, 0.1734, 2.5142000000000002, 0.16659859809715469, 3.2168000000000001, 0, 2, c_G4_STILBENE},
  {"G4_SUCROSE", 1.5804999999999998, 77.5, 1, true, 3.1526000000000001, 0.1341, 2.6558000000000002, 0.11300326565324842, 3.363, 0, 3, c_G4_SUCROSE},
  {"G4_TERPHENYL", 1.2399999999999998, 71.700000000000003, 1, true, 3.2639, 0.13220000000000001, 2.5428999999999999, 0.14963919604205167, 3.2685, 0, 2, c_G4_TERPHENYL},
  {"G4_TESTIS_ICRP", 1.0399999999999998, 75, 1, true, 3.4681044970113102, 0.23640958963753395, 2.0364095896375338, 0.40799007255330227, 3, 0, 9, c_G4_TESTIS_ICRP},
  {"G4_TETRACHLOROETHYLENE", 1.6250000000000002, 159.19999999999999, 1, true, 4.6619000000000002, 0.17130000000000001, 2.9083000000000001, 0.18595397628112253, 3.0156000000000001, 0, 2, c_G4_TETRACHLOROETHYLENE},
  {"G4_THALLIUM_CHLORIDE", 7.0039999999999987, 690.29999999999995, 1, true, 6.3008996638975878, 0.53082398751883419, 3.0464192636946854, 0.24224554867257761, 3, 0, 2, c_G4_THALLIUM_CHLORIDE},
  {"G4_TISSUE_SOFT_ICRP", 1.0299999999999998, 72.299999999999997, 1, true, 3.4354, 0.22109999999999999, 2.7799, 0.089268127824428778, 3.5110000000000001, 0, 9, c_G4_TISSUE_SOFT_ICRP},
  {"G4_TISSUE_SOFT_ICRU-4", 1, 74.900000000000006, 1, true, 3.5087000000000002, 0.23769999999999999, 2.7907999999999999, 0.096297192554213026, 3.4371, 0, 4, c_G4_TISSUE_SOFT_ICRU_4},
  {"G4_TISSUE-METHANE", 0.0010640899999999997, 61.200000000000003, 3, true, 9.9499999999999993, 1.6442000000000001, 4.1398999999999999, 0.099464765636123484, 3.4708000000000001, 0, 4, c_G4_TISSUE_METHANE},
  {"G4_TISSUE-PROPANE", 0.00182628, 59.5, 3, true, 9.3529, 1.5139, 3.9916, 0.098027455543409525, 3.5158999999999998, 0, 4, c_G4_TISSUE_PROPANE},
  {"G4_TITANIUM_DIOXIDE", 4.2599999999999998, 179.5, 1, true, 3.9521999999999999, -0.011900000000000001, 3.1646999999999998, 0.085692093697679397, 3.3267000000000002, 0, 2, c_G4_TITANIUM_DIOXIDE},
  {"G4_TOLUENE", 0.86689999999999989, 62.500000000000007, 1, true, 3.3026, 0.17219999999999999, 2.5728, 0.13283897976046313, 3.3557999999999999, 0, 2, c_G4_TOLUENE},
  {"G4_TRICHLOROETHYLENE", 1.4599999999999997, 148.09999999999999, 1, true, 4.6147999999999998, 0.18029999999999999, 2.9140000000000001, 0.18271368066627541, 3.0137, 0, 3, c_G4_TRICHLOROETHYLENE},
  {"G4_TRIETHYL_PHOSPHATE", 1.0700000000000001, 81.200000000000003, 1, true, 3.6242000000000001, 0.2054, 2.9428000000000001, 0.069220310338843974, 3.6301999999999999, 0, 4, c_G4_TRIETHYL_PHOSPHATE},
  {"G4_TUNGSTEN_HEXAFLUORIDE", 2.3999999999999995, 354.39999999999998, 1, true, 5.9881000000000002, 0.30199999999999999, 4.2602000000000002, 0.036581200296944422, 3.5133999999999999, 0, 2, c_G4_TUNGSTEN_HEXAFLUORIDE},
  {"G4_URANIUM_DICARBIDE", 11.279999999999998, 752, 1, true, 6.0247000000000002, -0.21909999999999999, 3.5207999999999999, 0.21119975033647834, 2.6577000000000002, 0, 2, c_G4_URANIUM_DICARBIDE},
  {"G4_URANIUM_MONOCARBIDE", 13.630000000000001, 862, 1, true, 6.1210000000000004, -0.25240000000000001, 3.4941, 0.22972575556151087, 2.6168999999999998, 0, 2, c_G4_URANIUM_MONOCARBIDE},
  {"G4_URANIUM_OXIDE", 10.960000000000001, 720.60000000000002, 1, true, 5.9604999999999997, -0.1938, 3.5291999999999999, 0.20462900896056799, 2.6711, 0, 2, c_G4_URANIUM_OXIDE},
  {"G4_UREA", 1.3229999999999997, 72.799999999999997, 1, true, 3.2031999999999998, 0.1603, 2.6524999999999999, 0.11609388978745998, 3.3460999999999999, 0, 4, c_G4_UREA},
  {"G4_VALINE", 1.23, 67.700000000000003, 1, true, 3.1059000000000001, 0.14410000000000001, 2.6227, 0.11386893730353781, 3.3774000000000002, 0, 4, c_G4_VALINE},
  {"G4_VITON", 1.8, 98.599999999999994, 1, true, 3.5943000000000001, 0.21060000000000001, 2.7873999999999999, 0.099657040810128564, 3.4556, 0, 3, c_G4_VITON},
  {"G4_WATER_VAPOR", 0.00075618199999999988, 71.599999999999994, 3, true, 10.5962, 1.7951999999999999, 4.3437000000000001, 0.081015240742794231, 3.5901000000000001, 0, 2, c_G4_WATER_VAPOR},
  {"G4_XYLENE", 0.86999999999999988, 61.799999999999997, 1, true, 3.2698, 0.16950000000000001, 2.5674999999999999, 0.13217159640260245, 3.3563999999999998, 0, 2, c_G4_XYLENE},
  {"G4_GRAPHITE", 2.21, 78, 1, true, 2.8679999999999999, -0.0178, 2.3414999999999999, 0.26141999999999999, 2.8696999999999999, 0.12, 1, c_G4_GRAPHITE},
  {"G4_lH2", 0.070800000000000002, 21.800000000000001, 2, true, 3.2631999999999999, 0.47589999999999999, 1.9215, 0.13482726522165447, 5.6249000000000002, 0, 1, c_G4_lH2},
  {"G4_lN2", 0.80700000000000005, 82, 2, true, 3.9996433678243779, 0.3038837379107473, 2, 0.53289430023438955, 3, 0, 1, c_G4_lN2},
  {"G4_lO2", 1.141, 95, 2, true, 3.9471004945707535, 0.2867547612300656, 2, 0.52230768204432199, 3, 0, 1, c_G4_lO2},
  {"G4_lAr", 1.3959999999999999, 188, 2, true, 5.2146147973013797, 0.20000000000000001, 3, 0.1955895025557472, 3, 0, 1, c_G4_lAr},
  {"G4_lBr", 3.1028000000000002, 343, 2, true, 5.6467704342129377, 0.34084716155341765, 3, 0.21683164399450608, 3, 0, 1, c_G4_lBr},
  {"G4_lKr", 2.4179999999999997, 352, 2, true, 5.967370225458243, 0.44536269349938701, 3.0000000000000004, 0.23490875653380347, 3, 0, 1, c_G4_lKr},
  {"G4_lXe", 2.9529999999999994, 482, 2, true, 6.4396525468296426, 0.59932673026646388, 2.9999999999999996, 0.26595456789349864, 3, 0, 1, c_G4_lXe},
  {"G4_PbWO4", 8.2799999999999994, 542.74149008633265, 1, true, 5.641544894762867, 0.33914363569269468, 3, 0.21655449646078637, 3, 0, 3, c_G4_PbWO4},
  {"G4_Galactic", 9.9999999999999992e-26, 21.800000000000001, 3, true, 105.21212437948088, 12.140989318008057, 13.303989318008057, 24.04850169063765, 4.7539999999999996, 0, 1, c_G4_Galactic},
  {"G4_GRAPHITE_POROUS", 1.6999999999999997, 78, 1, true, 3.1549999999999998, 0.048000000000000001, 2.5387, 0.20762, 2.9531999999999998, 0.14000000000000001, 1, c_G4_GRAPHITE_POROUS},
  {"G4_LUCITE", 1.1899999999999999, 74, 1, true, 3.3296999999999999, 0.18240000000000001, 2.6680999999999999, 0.11431670929235897, 3.3835999999999999, 0, 3, c_G4_LUCITE},
  {"G4_BRASS", 8.5199999999999996, 349.84539744383954, 1, true, 4.6455402748632935, 0.20000000000000001, 3, 0.16966591826100932, 3, 0, 3, c_G4_BRASS},
  {"G4_BRONZE", 8.8200000000000003, 339.60850221909504, 1, true, 4.5490106439811324, 0.20000000000000001, 3, 0.16526861364720821, 3, 0, 3, c_G4_BRONZE},
  {"G4_STAINLESS-STEEL", 8, 282.97693634121993, 1, true, 4.2532787147077933, 0.20000000000000001, 3, 0.15179686030931921, 3, 0, 3, c_G4_STAINLESS_STEEL},
  {"G4_CR39", 1.3200000000000001, 70.7753272857761, 1, true, 3.1500291724175109, 0.20000000000000001, 2, 0.38220081193756722, 3, 0, 3, c_G4_CR39},
  {"G4_OCTADECANOL", 0.81199999999999994, 55.759788903508337, 1, true, 3.0918023723563706, 0.20000000000000001, 2, 0.37221679272269409, 3, 0, 3, c_G4_OCTADECANOL},
  {"G4_KEVLAR", 1.4399999999999999, 71.862023137009984, 1, true, 3.1160010060272936, 0.20000000000000001, 2, 0.37636607833156288, 3, 0, 4, c_G4_KEVLAR},
  {"G4_DACRON", 1.3999999999999999, 74.266407050430971, 1, true, 3.2101900651778141, 0.20000000000000001, 2, 0.39251646570305132, 3, 0, 3, c_G4_DACRON},
  {"G4_NEOPRENE", 1.23, 90.094468323831947, 1, true, 3.7276051408656565, 0.21519927592220411, 2, 0.48132485631520933, 3, 0, 3, c_G4_NEOPRENE},
  {"G4_CYTOSINE", 1.3, 72, 1, true, 3.2191194824497549, 0.19804531797509387, 1.9980453179750939, 0.39559106458098769, 3, 0, 4, c_G4_CYTOSINE},
  {"G4_THYMINE", 1.48, 72, 1, true, 3.0869615560988346, 0.19521708626567297, 1.9952170862656731, 0.37516352034168238, 3, 0, 4, c_G4_THYMINE},
  {"G4_URACIL", 1.3200000000000001, 72, 1, true, 3.2126757913997399, 0.17835927233852125, 1.9783592723385213, 0.41003103361314314, 3, 0, 4, c_G4_URACIL},
  {"G4_DEOXYRIBOSE", 1.5, 72, 1, true, 3.0481580455651454, 0.2117351945285878, 2.0117351945285877, 0.35546663930001776, 3, 0, 3, c_G4_DEOXYRIBOSE},
  {"G4_PHOSPHORIC_ACID", 1.8699999999999999, 72, 1, true, 2.8784318370712154, 0.043177484385555104, 2.8431774843855551, 0.12206596999238756, 3, 0, 3, c_G4_PHOSPHORIC_ACID},
  {"G4_DNA_DEOXYRIBOSE", 1, 72, 1, true, 3.4449613290918593, 0.24471615145460113, 2.0447161514546011, 0.39746258648549088, 3, 0, 3, c_G4_DNA_DEOXYRIBOSE},
  {"G4_DNA_PHOSPHATE", 1, 72, 1, true, 3.5349027278245, -0.003472757104370866, 2.7965272428956292, 0.16175725971686991, 3, 0, 2, c_G4_DNA_PHOSPHATE},
  {"G4_DNA_ADENINE", 1, 72, 1, true, 3.496098060822725, 0.18951325253331688, 1.9895132525333168, 0.44982120720278834, 3, 0, 3, c_G4_DNA_ADENINE},
  {"G4_DNA_GUANINE", 1, 72, 1, true, 3.4990956999793528, 0.17809461758361256, 1.9780946175836125, 0.45935179643318746, 3, 0, 4, c_G4_DNA_GUANINE},
  {"G4_DNA_CYTOSINE", 1, 72, 1, true, 3.4897619043874517, 0.18794030131276676, 1.9879403013127668, 0.44997682305148812, 3, 0, 4, c_G4_DNA_CYTOSINE},
  {"G4_DNA_THYMINE", 1, 72, 1, true, 3.4862466850613041, 0.18631225270471846, 1.9863122527047183, 0.45065964568265693, 3, 0, 4, c_G4_DNA_THYMINE},
  {"G4_DNA_URACIL", 1, 72, 1, true, 3.498666110707652, 0.16790888645450081, 1.9679088864545007, 0.46732117847135207, 3, 0, 4, c_G4_DNA_URACIL},
};

inline constexpr int kNumNistMaterials =
    static_cast<int>(sizeof(kNistMaterials) / sizeof(kNistMaterials[0]));

}  // namespace g4gpu::g4::nist
