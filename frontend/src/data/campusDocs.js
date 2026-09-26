// CU Boulder campus notes used for retrieval on the hosted demo.
//
// The hackathon build indexed a CU Boulder information PDF on the AMD AI PC.
// That PDF is not in this repository, so the web demo retrieves over these
// short notes instead. They were summarized from public colorado.edu pages
// (linked per note) in September 2026. Campus details change: check the
// source link before relying on any of them.

const campusDocs = [
  {
    id: 'about',
    title: 'About CU Boulder',
    source: 'https://www.colorado.edu/about',
    text:
      'The University of Colorado Boulder was founded in 1876 in Boulder, Colorado. It is a Carnegie R1 research university and a member of the Association of American Universities (AAU). The athletic teams are the Buffaloes (Buffs). CU Boulder reports five Nobel laureates since 1989 and says it is the only university to have sent space instruments to every planet in the solar system. Undergraduate residents get a four-year tuition guarantee, and every applicant is automatically considered for scholarships. CU Promise supports qualifying Colorado residents who receive federal Pell Grants.',
  },
  {
    id: 'ralphie',
    title: 'Ralphie, the live buffalo mascot',
    source: 'https://cubuffs.com/sports/2016/6/15/Ralphie',
    text:
      'Ralphie is CU Boulder\'s live buffalo mascot, and Ralphie has always been a female bison because of size and temperament. On football game days Ralphie runs across Folsom Field in a horseshoe pattern, leading the team onto the field at the start of the game and again at the start of the second half, at up to 25 miles per hour. Five student Ralphie Handlers run with her: four beside her to guide her and one behind to control her speed. Handlers are varsity student-athletes who put in 20 to 30 hours a week during football season.',
  },
  {
    id: 'caps',
    title: 'Counseling and Psychiatric Services (CAPS)',
    source: 'https://www.colorado.edu/counseling/',
    text:
      'Counseling and Psychiatric Services (CAPS) offers mental health support for CU Boulder students: 20-minute screenings to find the right service, individual counseling, psychiatric services, therapy groups, workshops on topics such as anxiety, motivation and healthy habits, and referrals to community providers. Same-day screenings and appointments are available. CAPS is in the Center for Community (C4C), Suite N352, 2249 Willard Loop Dr. Phone 303-492-2277, answered 24/7, including for students in crisis. For a life-threatening emergency, call 911.',
  },
  {
    id: 'health',
    title: 'Medical Services at Wardenburg Health Center',
    source: 'https://www.colorado.edu/healthcenter/',
    text:
      'Medical Services for students is at Wardenburg Health Center, 1900 Wardenburg Drive. It offers primary care, sexual health, cold and flu care, immunizations (including flu and COVID-19 vaccines), physical therapy, nutrition counseling and a pharmacy. Book appointments through the MyCUHealth patient portal (mycuhealth.colorado.edu) or by calling 303-492-5101. After-hours options are listed on the clinic hours page. For a life-threatening emergency, dial 911.',
  },
  {
    id: 'bus',
    title: 'Riding the bus with your Buff OneCard (RTD)',
    source: 'https://www.colorado.edu/pts/transportation-options/bus/bus-pass-information',
    text:
      'Your Buff OneCard is your transit pass. Enrolled, tuition-paying CU Boulder students can ride regularly scheduled RTD service fare-free by tapping the Buff OneCard: local, limited, express and regional buses and light rail in all fare zones, including the SkyRide bus to Denver International Airport. The Stampede bus to East Campus is included too. Starting with students who entered in Fall 2025, the transit pass fee is folded into tuition instead of billed separately. A student taking a semester off can buy an optional transit pass at the Buff OneCard office. The Transit app shows real-time locations for Buff Buses, RTD and HOP.',
  },
  {
    id: 'parking',
    title: 'Parking on campus',
    source: 'https://www.colorado.edu/pts/',
    text:
      'All parking on the CU Boulder campus requires a permit. Your license plate is your permit once it is registered and the permit fee is paid, through the online parking portal (cuboulder.aimsparking.com). Parking and Transportation Services can be reached at 303-735-PARK (7275). Real-time availability for visitor lots is shown on the mPark tool.',
  },
  {
    id: 'onecard',
    title: 'Buff OneCard (student ID)',
    source: 'https://www.colorado.edu/buffonecard/',
    text:
      'The Buff OneCard is the CU Boulder student ID. Use it in dining centers and grab-and-go locations that accept Munch Money or Campus Cash, to pay for printing with Campus Cash, as your RTD transit pass, and optionally as an ATM card through Elevations Credit Union. The Buff OneCard Office is in the Center for Community (C4C), Room N180, 2249 Willard Loop Dr., open Monday to Friday, 8:00 a.m. to 4:30 p.m. Phone 303-492-0355, email boc@colorado.edu. Incoming first-year students can start the card process online before arriving.',
  },
  {
    id: 'pantry',
    title: 'Buff Pantry (free food)',
    source: 'https://www.colorado.edu/support/basicneeds/buff-pantry',
    text:
      'The Buff Pantry, run by the Basic Needs Center, gives CU Boulder undergraduate and graduate students experiencing food insecurity fresh produce, shelf-stable, refrigerated and frozen food and personal care items at no cost. It is in the Center for Community (C4C), Room N161. Students can visit once per week or place one online order per week (orders need 48 hours notice, excluding weekends); each student gets 20 credits a week. Bring your Buff OneCard and reusable bags. Phone 303-735-9863, email basicneeds@colorado.edu.',
  },
  {
    id: 'career',
    title: 'Career Services',
    source: 'https://www.colorado.edu/career/',
    text:
      'Career Services helps with career advising, resumes, cover letters, interview practice, major exploration, and skill-building workshops, and runs career fairs and networking events. Students search and apply for jobs and internships on Handshake. Scheduled, drop-in and express appointments are available through the Career Services website. Career Services is in the Center for Community (C4C), S440. Phone 303-492-6541.',
  },
  {
    id: 'tutoring',
    title: 'Free tutoring and academic help',
    source: 'https://www.colorado.edu/today/2023/09/05/6-tutoring-resources-explore',
    text:
      'The Academic Success and Achievement Program (ASAP) offers free one-hour tutoring with trained peer tutors in hundreds of courses for first-year students and students living on campus. The Mathematics Academic Resource Center (MARC) is a free help center for CU math courses. The Writing Center offers free one-on-one sessions with trained writing consultants for any discipline. ALTEC offers free language tutoring for students in the first three semesters of ASL, French, German, Italian, Japanese, Russian and Spanish. The International Student Tutoring Program helps international students with English and U.S. academic culture.',
  },
  {
    id: 'libraries',
    title: 'University Libraries',
    source: 'https://libraries.colorado.edu/libraries-collections/norlin-library',
    text:
      'Norlin Library is the main and largest library on campus, at 1720 Pleasant Street. It holds the humanities, social sciences and life sciences collections, Rare and Distinctive Collections, and Norlin Commons. There are four branch libraries: the Jerry Crail Johnson Earth Sciences and Map Library; the Gemmill Engineering, Mathematics and Physics Library; the Waltz Music Library; and the William M. White Business Library. The University Libraries hold the largest library collection in the Rocky Mountain region.',
  },
  {
    id: 'rec',
    title: 'Student Recreation Center',
    source: 'https://www.colorado.edu/recreation/',
    text:
      'The Student Recreation Center has fitness spaces, pools and an ice rink, and runs fitness and wellness programs, intramural sports, sport clubs, personal training, CPR and lifeguard training, outdoor pursuits and inclusive recreation. During renovations the main entrance moved to the east side at the top of Stadium Drive; on football game days use the north entrance. Passes and class registration are through the recreation portal. Phone 303-492-6880.',
  },
  {
    id: 'cs',
    title: 'Computer Science degrees',
    source: 'https://www.colorado.edu/cs/academics/undergraduate-programs/bachelor-science',
    text:
      'Computer Science at CU Boulder is in the College of Engineering and Applied Science. It offers a Bachelor of Science (BS) and a Bachelor of Arts (BA). The BS covers a wider range of CS courses and more mathematics than the BA, from circuits and computer architecture through operating systems and programming languages to large software systems, and seniors complete a year-long capstone project. Students can follow suggested plans of study in areas such as Artificial Intelligence and Machine Learning, Robotics, Software Engineering, and Systems, Networks and Security.',
  },
];

export default campusDocs;
