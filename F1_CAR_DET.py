import streamlit as st
from PIL import Image
import tensorflow as tf

# Define class names globally
class_names = ["AlphaTauri F1 car", "Aston Martin F1 Car", "Ferrari F1 car", "Lotus Formula 1 Car",
               "McLaren F1 car", "Mercedes F1 car", "Racing Point F1 car", "Red Bull Racing F1 car",
               "Renault F1 car", "Williams F1 car"]


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))  # Resize to model input size
    img = tf.convert_to_tensor(img)  # Convert to TensorFlow tensor
    img = tf.cast(img, dtype=tf.float32)  # Convert to float32
    img = img / 255.0  # Normalize pixel values
    img = tf.expand_dims(img, axis=0)  # Add batch dimension
    return img


# Function to make predictions using your loaded model
def make_prediction(model, image):
    predictions = model.predict(image)  # Pass the preprocessed image to your model
    sorted_indices = tf.argsort(predictions[0])[::-1]  # Sort indices of predictions in descending order
    predicted_class = class_names[sorted_indices[0]]
    second_highest_class = class_names[sorted_indices[1]]
    return predicted_class, second_highest_class, predictions[0], sorted_indices

def get_car_info(car_name):
    car_info = {
        "AlphaTauri F1 car": "Scuderia AlphaTauri, or simply AlphaTauri, was an Italian Formula One racing team and constructor that competed from 2020 to 2023. It was one of two Formula One constructors owned by Austrian conglomerate Red Bull GmbH, the other being Red Bull Racing. The constructor was rebranded for the 2020 Formula One World Championship from Toro Rosso to AlphaTauri in order to promote Red Bull's AlphaTauri fashion brand.[3] According to Franz Tost and Helmut Marko, the rebrand as Scuderia AlphaTauri also acknowledged that it had transitioned from Red Bull Racing's junior team to its sister team.[4] Throughout its history, the team has only scored one victory and two podiums, all scored by Pierre Gasly, winning the 2020 Italian Grand Prix and placing third in the 2021 Azerbaijan Grand Prix. The team was rebranded as RB in 2024",
        "Aston Martin F1 Car": "Aston Martin is a British car manufacturer that has participated in Formula One in various forms and is currently represented by a team named as Aston Martin Aramco F1 Team. The company first participated in Formula One during the 1959 season, where they debuted the DBR4 chassis using their own engine, but it failed to score any points. They continued to perform poorly through the 1960 season, once again failing to score any points. As a result, Aston Martin decided to leave Formula One after 1960. A commercial rebranding of the Racing Point F1 Team resulted in the team's return as Aston Martin in 2021, utilising customer Mercedes power units. In 2026, the team will begin using Honda power units as part of a works partnership with the Japanese manufacturer. The team, owned by Lawrence Stroll, has Fernando Alonso and Lance Stroll as their race drivers beginning with the 2023 season. The team is headquartered in Silverstone and has previously raced under various different names, starting with Jordan Grand Prix in 1991.",
        "Ferrari F1 car": "Scuderia Ferrari (Italian: [skudeˈriːa ferˈraːri]) is the racing division of luxury Italian auto manufacturer Ferrari and the racing team that competes in Formula One racing. The team is also known by the nickname 'The Prancing Horse' (Italian: il Cavallino Rampante or simply il Cavallino), in reference to their logo. It is the oldest surviving and most successful Formula One team, having competed in every world championship since 1950. The team was founded by Enzo Ferrari, initially to race cars produced by Alfa Romeo. By 1947, Ferrari had begun building its own cars. Among its important achievements outside Formula One are winning the World Sportscar Championship, 24 Hours of Le Mans, 24 Hours of Spa, 24 Hours of Daytona, 12 Hours of Sebring, Bathurst 12 Hour, races for Grand tourer cars, and racing on road courses of the Targa Florio, the Mille Miglia, and the Carrera Panamericana. The team is also known for its passionate support base, known as the tifosi. The Italian Grand Prix at Monza is regarded as the team's home race. As a constructor in Formula One, Ferrari has a record 16 Constructors' Championships. Their most recent Constructors' Championship was won in 2008. The team also holds the record for the most Drivers' Championships with 15, won by nine different drivers including Alberto Ascari, Juan Manuel Fangio, Mike Hawthorn, Phil Hill, John Surtees, Niki Lauda, Jody Scheckter, Michael Schumacher, and Kimi Räikkönen. Räikkönen's title in 2007 is the most recent for the team. The 2020 Tuscan Grand Prix marked Ferrari's 1000th Grand Prix in Formula One. Schumacher is the team's most successful driver. Joining the team in 1996 and driving for them until his first retirement in 2006, he won five consecutive drivers' titles and 72 Grands Prix for the team. His titles came consecutively between 2000 and 2004, and the team won consecutive constructors' titles between 1999 and 2004, marking the era as the most successful period in the team's history. The team's drivers for the 2024 season are Charles Leclerc and Carlos Sainz Jr. The latter will be replaced by Lewis Hamilton in 2025.",
        "Lotus Formula 1 Car": "Lotus F1 Team was a British Formula One racing team. The team competed under the Lotus name from 2012 until 2015, following the renaming of the former Renault team based at Enstone in Oxfordshire. The Lotus F1 Team was majority owned by Genii Capital.[1][2] Lotus F1 was named after its branding partner Group Lotus. The team achieved a race victory and fourth position in the Formula One Constructors' World Championship in their first season under the Lotus title. Lotus F1 achieved 2 race victories in their time on the grid, both courtesy of Kimi Räikkönen. The team was sold back to Renault on 18 December 2015. The Lotus F1 Team name was officially dropped on 3 February 2016, as Renault announced that the team would compete as Renault Sport Formula One Team.",
        "McLaren F1 car": "McLaren Racing Limited is a British motor racing team based at the McLaren Technology Centre in Woking, Surrey, England. McLaren is best known as a Formula One chassis constructor, the second-oldest active team and the second-most successful Formula One team after Ferrari, having won 184 races, 12 Drivers' Championships, and eight Constructors' Championships. McLaren also has a history in American open wheel racing as both an entrant and a chassis constructor, and has won the Canadian-American Challenge Cup (Can-Am) sports car racing championship. McLaren is also one of only three constructors to complete the Triple Crown of Motorsport (wins at the Indianapolis 500, 24 Hours of Le Mans, and Monaco Grand Prix), a feat that McLaren achieved by winning the 1995 24 Hours of Le Mans. The team is a subsidiary of the McLaren Group, which owns a majority of the team.",
        "Mercedes F1 car": "Mercedes-Benz, a German luxury automotive brand of the Mercedes-Benz Group, has been involved in Formula One as both team owner and engine manufacturer for various periods since 1954. The current Mercedes-AMG Petronas F1 Team is based in Brackley, England,[4] and possesses a German licence.[5] An announcement was made in December 2020 that Ineos planned to take a one third equal ownership stake alongside the Mercedes-Benz Group and Toto Wolff;[6] this came into effect on 25 January 2022.[7] Mercedes-branded teams are often referred to by the nickname, the 'Silver Arrows' (German: Silberpfeile). Before the Second World War, Mercedes-Benz competed in the European Championship, winning three titles. The marque debuted in Formula One in 1954. After winning their first race at the 1954 French Grand Prix, driver Juan Manuel Fangio won another three Grands Prix to win the 1954 Drivers' Championship and repeated this success in 1955. Despite winning two Drivers' Championships, Mercedes-Benz withdrew from motor racing after 1955 in response to the 1955 Le Mans disaster.",
        "Racing Point F1 car": "Racing Point F1 Team, which competed as BWT Racing Point F1 Team and commonly known as Racing Point, was a British motor racing team and constructor that Racing Point UK entered into the Formula One World Championship. The team was based in Silverstone, England and competed under a British licence. The team was renamed in February 2019 from Racing Point Force India F1 Team, which used the constructor name of Force India for the latter half of the 2018 season. Racing Point made their racing debut at the 2019 Australian Grand Prix. The team's drivers for the 2020 season were Sergio Pérez and Lance Stroll. The team rebranded to Aston Martin for the 2021 Formula One season.",
        "Red Bull Racing F1 car": "Red Bull Racing, currently competing as Oracle Red Bull Racing and also known simply as Red Bull or RBR, is a Formula One racing team, racing under an Austrian licence and based in the United Kingdom. It is one of two Formula One teams owned by conglomerate Red Bull GmbH, the other being RB Formula One Team. The Red Bull Racing team has been managed by Christian Horner since its formation in 2005. Red Bull had Cosworth engines in 2005 and Ferrari engines in 2006. The team used engines supplied by Renault between 2007 and 2018 (from 2016 to 2018, the Renault engine was re-badged TAG Heuer following the breakdown in the relationship between Red Bull and Renault in 2015).[9][10] During this partnership, they won four successive Drivers' and Constructors' Championship titles from 2010 to 2013, becoming the first Austrian team to win the title.[11] The team began using Honda engines in 2019.[12] The works Honda partnership culminated in 2021 following Red Bull driver Max Verstappen's World Drivers' Championship victory, with Verstappen also winning the championship in 2022 and 2023. Honda left the sport officially after 2021 but is set to continue to supply complete engines from Japan to the team partly under Red Bull Powertrains branding until the end of 2025.",
        "Renault F1 car": "Renault, a French automobile manufacturer, has been associated with Formula One as both team owner and engine manufacturer for various periods since 1977.[1] In 1977, the company entered Formula One as a constructor, introducing the turbo engine to Formula One with its EF1 engine. In 1983, Renault began supplying engines to other teams.[2] Although the Renault team had won races, it withdrew at the end of 1985.[3] Renault engines continued to be raced until 1986. Renault returned to Formula One in 1989 as an engine manufacturer. It won five drivers' titles and six constructors' titles between 1992 and 1997 with Williams and Benetton, before ending its works involvement after 1997, though their engines continued to be used without works backing until 2000.",
        "Williams F1 car": "Williams Grand Prix Engineering Limited, currently racing in Formula One as Williams Racing, is a British Formula One team and constructor. It was founded by Sir Frank Williams (1942–2021) and Sir Patrick Head. The team was formed in 1977 after Frank Williams's earlier unsuccessful F1 operation: Frank Williams Racing Cars (which later became Wolf–Williams Racing in 1976). The team is based in Grove, Oxfordshire, on a 60-acre (24 ha) site. Williams FW36. The team's first race was the 1977 Spanish Grand Prix, where the new team ran a March chassis for Patrick Nève. Williams started manufacturing its own cars the following year, and Clay Regazzoni won Williams's first race at the 1979 British Grand Prix. At the 1997 British Grand Prix, Jacques Villeneuve scored the team's 100th race victory, making Williams one of only five teams in Formula One, alongside Ferrari, McLaren, Mercedes, and Red Bull Racing to win 100 races. Williams won nine Constructors' Championships between 1980 and 1997. This was a record until Ferrari won its tenth championship in 2000."
    }
    return car_info.get(car_name, "Information not available.")


def main():
    st.title("F1 Car Image Classification App")

    st.write("Upload an image to classify it. The app supports the following car brands:")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        - AlphaTauri F1 car
        - Aston Martin F1 Car
        - Ferrari F1 car
        - Lotus Formula 1 Car
        - McLaren F1 car
        """)
    with col2:
        st.write("""
        - Mercedes F1 car
        - Racing Point F1 car
        - Red Bull Racing F1 car
        - Renault F1 car
        - Williams F1 car
        """)

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            with st.spinner("Loading and processing the image..."):
                # Preprocess the image
                image = preprocess_image(uploaded_file)

                # Load your pre-trained model (replace with your model loading logic)
                model = tf.keras.models.load_model("F1_new_car_detection_10.h5")  # Replace with your model path

                # Make prediction and display results
                predicted_class, second_highest_class, probabilities, sorted_indices = make_prediction(model, image)
                car_info = get_car_info(predicted_class)

                st.subheader(
                    f"The car shown in the image is   : {predicted_class} ({probabilities[sorted_indices[0]] * 100:.2f}%)")
                st.write(
                    f"The second most probable car is   : {second_highest_class} ({probabilities[sorted_indices[1]] * 100:.2f}%)")
                st.image(uploaded_file)
                st.subheader("Here Is The Info About : "+predicted_class)
                st.write(car_info)



        except Exception as e:
            st.error(f"Error processing image: {e}")


main()
