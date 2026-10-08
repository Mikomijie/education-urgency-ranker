import numpy as np
import pandas as pd

np.random.seed(42)

NIGERIA_LGAS = {
    'Abia': ['Aba North','Aba South','Arochukwu','Bende','Ikwuano','Isiala Ngwa North','Isiala Ngwa South','Isuikwuato','Obi Ngwa','Ohafia','Osisioma Ngwa','Ugwunagbo','Ukwa East','Ukwa West','Umuahia North','Umuahia South','Umu Nneochi'],
    'Adamawa': ['Demsa','Fufore','Ganye','Girei','Gombi','Guyuk','Hong','Jada','Lamurde','Madagali','Maiha','Mayo-Belwa','Michika','Mubi North','Mubi South','Numan','Shelleng','Song','Toungo','Yola North','Yola South'],
    'Akwa Ibom': ['Abak','Eastern Obolo','Eket','Esit Eket','Essien Udim','Etim Ekpo','Etinan','Ibeno','Ibesikpo Asutan','Ibiono Ibom','Ika','Ikono','Ikot Abasi','Ikot Ekpene','Ini','Itu','Mbo','Mkpat Enin','Nsit Atai','Nsit Ibom','Nsit Ubium','Obot Akara','Okobo','Onna','Oron','Oruk Anam','Udung Uko','Ukanafun','Uruan','Urue-Offong/Oruko','Uyo'],
    'Anambra': ['Aguata','Anambra East','Anambra West','Anaocha','Awka North','Awka South','Ayamelum','Dunukofia','Ekwusigo','Idemili North','Idemili South','Ihiala','Njikoka','Nnewi North','Nnewi South','Ogbaru','Onitsha North','Onitsha South','Orumba North','Orumba South','Oyi'],
    'Bauchi': ['Alkaleri','Bauchi','Bogoro','Damban','Darazo','Dass','Gamawa','Ganjuwa','Giade','Itas/Gadau',"Jama'are",'Katagum','Kirfi','Misau','Ningi','Shira','Tafawa Balewa','Toro','Warji','Zaki'],
    'Bayelsa': ['Brass','Ekeremor','Kolokuma/Opokuma','Nembe','Ogbia','Sagbama','Southern Ijaw','Yenagoa'],
    'Benue': ['Ado','Agatu','Apa','Buruku','Gboko','Guma','Gwer East','Gwer West','Katsina-Ala','Konshisha','Kwande','Logo','Makurdi','Obi','Ogbadibo','Ohimini','Oju','Okpokwu','Otukpo','Tarka','Ukum','Ushongo','Vandeikya'],
    'Borno': ['Abadam','Askira/Uba','Bama','Bayo','Biu','Chibok','Damboa','Dikwa','Gubio','Guzamala','Gwoza','Hawul','Jere','Kaga','Kala/Balge','Konduga','Kukawa','Kwaya Kusar','Mafa','Magumeri','Maiduguri','Marte','Mobbar','Monguno','Ngala','Nganzai','Shani'],
    'Cross River': ['Abi','Akamkpa','Akpabuyo','Bakassi','Bekwarra','Biase','Boki','Calabar Municipal','Calabar South','Etung','Ikom','Obanliku','Obubra','Obudu','Odukpani','Ogoja','Yakurr','Yala'],
    'Delta': ['Aniocha North','Aniocha South','Bomadi','Burutu','Ethiope East','Ethiope West','Ika North East','Ika South','Isoko North','Isoko South','Ndokwa East','Ndokwa West','Okpe','Oshimili North','Oshimili South','Patani','Sapele','Udu','Ughelli North','Ughelli South','Ukwuani','Uvwie','Warri North','Warri South','Warri South West'],
    'Ebonyi': ['Abakaliki','Afikpo North','Afikpo South','Ebonyi','Ezza North','Ezza South','Ikwo','Ishielu','Ivo','Izzi','Ohaukwu','Ohaozara','Onicha'],
    'Edo': ['Akoko-Edo','Egor','Esan Central','Esan North-East','Esan South-East','Esan West','Etsako Central','Etsako East','Etsako West','Igueben','Ikpoba-Okha','Oredo','Orhionmwon','Ovia North-East','Ovia South-West','Owan East','Owan West','Uhunmwonde'],
    'Ekiti': ['Ado-Ekiti','Efon','Ekiti East','Ekiti South West','Ekiti West','Emure','Gbonyin','Ido-Osi','Ijero','Ikere','Ikole','Ilejemeje','Irepodun/Ifelodun','Ise/Orun','Moba','Oye'],
    'Enugu': ['Aninri','Awgu','Enugu East','Enugu North','Enugu South','Ezeagu','Igbo-Etiti','Igbo-Eze North','Igbo-Eze South','Isi-Uzo','Nkanu East','Nkanu West','Nsukka','Oji-River','Udenu','Udi','Uzo-Uwani'],
    'FCT': ['Abaji','Abuja Municipal','Bwari','Gwagwalada','Kuje','Kwali'],
    'Gombe': ['Akko','Balanga','Billiri','Dukku','Funakaye','Gombe','Kaltungo','Kwami','Nafada','Shongom','Yamaltu/Deba'],
    'Imo': ['Aboh Mbaise','Ahiazu Mbaise','Ehime Mbano','Ezinihitte Mbaise','Ideato North','Ideato South','Ihitte/Uboma','Ikeduru','Isiala Mbano','Isu','Mbaitoli','Ngor Okpala','Njaba','Nkwerre','Nwangele','Obowo','Oguta','Ohaji/Egbema','Okigwe','Onuimo','Orlu','Orsu','Oru East','Oru West','Owerri Municipal','Owerri North','Owerri West'],
    'Jigawa': ['Auyo','Babura','Biriniwa','Birnin Kudu','Buji','Dutse','Gagarawa','Garki','Gumel','Guri','Gwaram','Gwiwa','Hadejia','Jahun','Kafin Hausa','Kaugama','Kazaure','Kiri Kasama','Kiyawa','Maigatari','Malam Madori','Miga','Ringim','Roni','Sule Tankarkar','Taura','Yankwashi'],
    'Kaduna': ['Birnin Gwari','Chikun','Giwa','Igabi','Ikara','Jaba',"Jema'a",'Kachia','Kaduna North','Kaduna South','Kagarko','Kajuru','Kaura','Kauru','Kubau','Kudan','Lere','Makarfi','Sabon Gari','Sanga','Soba','Zangon Kataf','Zaria'],
    'Kano': ['Ajingi','Albasu','Bagwai','Bebeji','Bichi','Bunkure','Dala','Dambatta','Dawakin Kudu','Dawakin Tofa','Doguwa','Fagge','Gabasawa','Garko','Garun Mallam','Gaya','Gezawa','Gwale','Gwarzo','Kabo','Kano Municipal','Karaye','Kibiya','Kiru','Kumbotso','Kunchi','Kura','Madobi','Makoda','Minjibir','Nassarawa','Rano','Rimin Gado','Rogo','Shanono','Sumaila','Takai','Tarauni','Tofa','Tsanyawa','Tudun Wada','Ungogo','Warawa','Wudil'],
    'Katsina': ['Bakori','Batagarawa','Batsari','Baure','Bindawa','Charanchi','Dan Musa','Dandume','Danja','Daura','Dutsi','Dutsin-Ma','Faskari','Funtua','Ingawa','Jibia','Kafur','Kaita','Kankara','Kankia','Katsina','Kurfi','Kusada',"Mai'Adua",'Malumfashi','Mani','Mashi','Matazu','Musawa','Rimi','Sabuwa','Safana','Sandamu','Zango'],
    'Kebbi': ['Aleiro','Arewa Dandi','Argungu','Augie','Bagudo','Birnin Kebbi','Bunza','Dandi','Fakai','Gwandu','Jega','Kalgo','Koko/Besse','Maiyama','Ngaski','Sakaba','Shanga','Suru','Wasagu/Danko','Yauri','Zuru'],
    'Kogi': ['Adavi','Ajaokuta','Ankpa','Bassa','Dekina','Ibaji','Idah','Igalamela-Odolu','Ijumu','Kabba/Bunu','Kogi','Lokoja','Mopa-Muro','Ofu','Ogori/Magongo','Okehi','Okene','Olamaboro','Omala','Yagba East','Yagba West'],
    'Kwara': ['Asa','Baruten','Edu','Ekiti','Ifelodun','Ilorin East','Ilorin South','Ilorin West','Irepodun','Isin','Kaiama','Moro','Offa','Oke Ero','Oyun','Patigi'],
    'Lagos': ['Agege','Ajeromi-Ifelodun','Alimosho','Amuwo-Odofin','Apapa','Badagry','Epe','Eti-Osa','Ibeju-Lekki','Ifako-Ijaiye','Ikeja','Ikorodu','Kosofe','Lagos Island','Lagos Mainland','Mushin','Ojo','Oshodi-Isolo','Somolu','Surulere'],
    'Nasarawa': ['Akwanga','Awe','Doma','Karu','Keana','Keffi','Kokona','Lafia','Nasarawa','Nasarawa Eggon','Obi','Toto','Wamba'],
    'Niger': ['Agaie','Agwara','Bida','Borgu','Bosso','Chanchaga','Edati','Gbako','Gurara','Katcha','Kontagora','Lapai','Lavun','Magama','Mariga','Mashegu','Mokwa','Munya','Paikoro','Rafi','Rijau','Shiroro','Suleja','Tafa','Wushishi'],
    'Ogun': ['Abeokuta North','Abeokuta South','Ado-Odo/Ota','Ewekoro','Ifo','Ijebu East','Ijebu North','Ijebu North East','Ijebu Ode','Ikenne','Imeko Afon','Ipokia','Obafemi Owode','Odeda','Odogbolu','Ogun Waterside','Remo North','Sagamu','Yewa North','Yewa South'],
    'Ondo': ['Akoko North-East','Akoko North-West','Akoko South-East','Akoko South-West','Akure North','Akure South','Ese Odo','Idanre','Ifedore','Ilaje','Ile Oluji/Okeigbo','Irele','Odigbo','Okitipupa','Ondo East','Ondo West','Ose','Owo'],
    'Osun': ['Aiyedaade','Aiyedire','Atakunmosa East','Atakunmosa West','Boluwaduro','Boripe','Ede North','Ede South','Egbedore','Ejigbo','Ife Central','Ife East','Ife North','Ife South','Ifedayo','Ifelodun','Ila','Ilesa East','Ilesa West','Irepodun','Irewole','Isokan','Iwo','Obokun','Odo Otin','Ola Oluwa','Olorunda','Oriade','Orolu','Osogbo'],
    'Oyo': ['Afijio','Akinyele','Atiba','Atisbo','Egbeda','Ibadan North','Ibadan North-East','Ibadan North-West','Ibadan South-East','Ibadan South-West','Ibarapa Central','Ibarapa East','Ibarapa North','Ido','Irepo','Iseyin','Itesiwaju','Iwajowa','Kajola','Lagelu','Ogbomosho North','Ogbomosho South','Ogo Oluwa','Olorunsogo','Oluyole','Ona Ara','Orelope','Ori Ire','Oyo East','Oyo West','Saki East','Saki West','Surulere'],
    'Plateau': ['Barkin Ladi','Bassa','Bokkos','Jos East','Jos North','Jos South','Kanke','Kanam','Langtang North','Langtang South','Mangu','Mikang','Pankshin',"Qua'an Pan",'Riyom','Shendam','Wase'],
    'Rivers': ['Abua-Odual','Ahoada East','Ahoada West','Akuku-Toru','Andoni','Asari-Toru','Bonny','Degema','Eleme','Emohua','Etche','Gokana','Ikwerre','Khana','Obio-Akpor','Ogba-Egbema-Ndoni','Ogu-Bolo','Okrika','Omuma','Opobo-Nkoro','Oyigbo','Port Harcourt','Tai'],
    'Sokoto': ['Binji','Bodinga','Dange/Shuni','Gada','Goronyo','Gudu','Gwadabawa','Illela','Isa','Kebbe','Kware','Rabah','Sabon Birni','Shagari','Silame','Sokoto North','Sokoto South','Tambuwal','Tangaza','Tureta','Wamakko','Wurno','Yabo'],
    'Taraba': ['Ardo Kola','Bali','Donga','Gashaka','Gassol','Ibi','Jalingo','Karim Lamido','Kurmi','Lau','Sardauna','Takum','Ussa','Wukari','Yorro','Zing'],
    'Yobe': ['Bade','Bursari','Damaturu','Fika','Fune','Geidam','Gujba','Gulani','Jakusko','Karasuwa','Machina','Nangere','Nguru','Potiskum','Tarmuwa','Yunusari','Yusufari'],
    'Zamfara': ['Anka','Bakura','Birnin Magaji','Bukkuyum','Bungudu','Gummi','Gusau','Kauran Namoda','Maradun','Maru','Shinkafi','Talatan Mafara','Tsafe','Zurmi'],
}

# Geopolitical zones — used for zone-level analysis in the app
ZONE_MAP = {
    'Kano': 'North West', 'Katsina': 'North West', 'Sokoto': 'North West',
    'Kebbi': 'North West', 'Zamfara': 'North West', 'Kaduna': 'North West', 'Jigawa': 'North West',
    'Borno': 'North East', 'Yobe': 'North East', 'Adamawa': 'North East',
    'Gombe': 'North East', 'Bauchi': 'North East', 'Taraba': 'North East',
    'Niger': 'North Central', 'Kogi': 'North Central', 'Kwara': 'North Central',
    'Nasarawa': 'North Central', 'Benue': 'North Central', 'Plateau': 'North Central', 'FCT': 'North Central',
    'Lagos': 'South West', 'Ogun': 'South West', 'Oyo': 'South West',
    'Osun': 'South West', 'Ondo': 'South West', 'Ekiti': 'South West',
    'Rivers': 'South South', 'Delta': 'South South', 'Edo': 'South South',
    'Bayelsa': 'South South', 'Cross River': 'South South', 'Akwa Ibom': 'South South',
    'Enugu': 'South East', 'Anambra': 'South East', 'Imo': 'South East',
    'Abia': 'South East', 'Ebonyi': 'South East',
}

# North tends to have worse infrastructure and higher dropout
ZONE_PROFILES = {
    'North West':  {'facility_base': 0.35, 'dropout_base': 0.28},
    'North East':  {'facility_base': 0.30, 'dropout_base': 0.32},
    'North Central': {'facility_base': 0.50, 'dropout_base': 0.18},
    'South West':  {'facility_base': 0.72, 'dropout_base': 0.08},
    'South South': {'facility_base': 0.65, 'dropout_base': 0.12},
    'South East':  {'facility_base': 0.68, 'dropout_base': 0.10},
}

rows = []
for state, lga_list in NIGERIA_LGAS.items():
    zone = ZONE_MAP.get(state, 'North Central')
    profile = ZONE_PROFILES[zone]

    for lga_name in lga_list:
        enrollment = np.random.randint(3000, 60000)
        teachers = np.random.randint(80, 1500)
        classrooms = np.random.randint(30, 800)

        student_teacher_ratio = round(enrollment / teachers, 2)
        pupil_classroom_ratio = round(enrollment / classrooms, 2)

        # ── Facility Score (0-1): electricity, water, toilets, building condition ──
        facility_score = round(
            np.clip(
                np.random.beta(
                    a=max(0.5, profile['facility_base'] * 5),
                    b=max(0.5, (1 - profile['facility_base']) * 5)
                ) + np.random.normal(0, 0.05),
                0.05, 0.99
            ), 2
        )

        # ── Grade Dropout Rate: fraction of enrolled pupils who drop before JSS3 ──
        dropout_rate = round(
            np.clip(
                profile['dropout_base'] + np.random.normal(0, 0.06)
                + (0.05 if student_teacher_ratio > 55 else 0)
                + (0.04 if facility_score < 0.4 else 0),
                0.02, 0.65
            ), 3
        )

        # ── Pass rate: weakly correlated with resources + noise ──
        base_pass = (
            60
            - (student_teacher_ratio * 0.4)
            - (pupil_classroom_ratio * 0.05)
            + (facility_score * 10)        # better facilities → modest boost
            - (dropout_rate * 20)          # high dropout → lower pass rate
            + np.random.normal(0, 12)
        )
        base_pass = np.clip(base_pass, 10, 95)

        if student_teacher_ratio > 60:
            base_pass -= np.random.uniform(2, 8)
        if student_teacher_ratio < 30:
            base_pass += np.random.uniform(1, 5)
        if pupil_classroom_ratio > 80:
            base_pass -= np.random.uniform(1, 5)
        if facility_score < 0.3:
            base_pass -= np.random.uniform(3, 10)

        # Inject real-world anomalies
        anomaly_type = np.random.choice(
            ['underperform_severe', 'underperform_mild', 'normal', 'overperform'],
            p=[0.08, 0.18, 0.62, 0.12]
        )
        if anomaly_type == 'underperform_severe':
            base_pass -= np.random.uniform(15, 30)  # ghost teachers, resource diversion
        elif anomaly_type == 'underperform_mild':
            base_pass -= np.random.uniform(5, 15)
        elif anomaly_type == 'overperform':
            base_pass += np.random.uniform(8, 20)   # high-efficiency star LGA

        actual_pass_rate = round(np.clip(base_pass, 5, 98), 1)

        rows.append({
            'State': state,
            'Zone': zone,
            'LGA': lga_name,
            'Enrollment': enrollment,
            'Teachers': teachers,
            'Classrooms': classrooms,
            'StudentTeacherRatio': student_teacher_ratio,
            'PupilClassroomRatio': pupil_classroom_ratio,
            'FacilityScore': facility_score,
            'GradeDropoutRate': dropout_rate,
            'ActualPassRate': actual_pass_rate,
        })

df = pd.DataFrame(rows)
df.to_csv('education_data.csv', index=False)
print(f"Generated {len(df)} LGAs across {df['State'].nunique()} states and {df['Zone'].nunique()} zones")
print(f"New columns: FacilityScore (mean={df['FacilityScore'].mean():.2f}), GradeDropoutRate (mean={df['GradeDropoutRate'].mean():.2f})")
print(df[['StudentTeacherRatio', 'FacilityScore', 'GradeDropoutRate', 'ActualPassRate']].corr())