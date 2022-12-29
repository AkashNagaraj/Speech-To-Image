import whisper
from pyChatGPT import ChatGPT

session_token ="eyJhbGciOiJkaXIiLCJlbmMiOiJBMjU2R0NNIn0..6MrD_VzX2rPmjOSW.HAki--JiWyB7M9K4eZKFZfOJzlckBuPVvBcWXogTlORKH3K_kt-02fQTv5WexCeBqyFJg3vo_ojKm1KV-DqjxsBC9zRGqNL7AC2iMAcbEamGMeZlHQkoG05jhqLE3k5xqtV7qOWl7Jp9TlJqMIrtMlL9laYvTQRF_m_tzP6WmKksSYnUzPVq_YWUhy7i1Qlnd-V0Q8_epufjHyX7zVkZq88ODXQw-oc6g-wgUOpzzDG14l0CkjIhWv90HSGATl6XeOHISbvoYv7vKcRQb2tAAcXnGM7rtNt76WI9Yh_JkdoOlHOsSwff0tBIYkvKHlgHDaA1jGPRkxDWTT7Eht_pMwbertaImwbmcASnelZ1OD5tQy92OJUqqyI41OtGwDw90nI8RMzUpNcYtYEIczVSHMEv5fFUjAUrqKRpSkZXFMr7LqJJy4-SQ8QbF_9NsmR6Mgjfup1UW2n6Dsg2z-71sFWSuvClAEX-iLAm2qs0nOIOlkM-Qbwd0MGgpJNZBtcScEcdoy5DL6oIHaqiLCRG_rlppaE4Y-1hwIYqaN_rUEFNLAQYAR_4OB0hzTbNsuW-mKZy-YgM4NenWPWMDT0aX9DAa6-VFOK7hbs9vQ2vla2aajFqOj9UqBK-ytvoQvTEi5CZM1C6PnA1sIsSV7kjiRmezQD5SdNiBUwJCK_oPQGGBU1-qQJXvnZ2YlY9_abIEXn2YyWNCZYWOBnhounDq3yyYXVTXcoT1l5aTj6YhpAU7Y99eP3g_ZbHTlMC_Zj7lEOWD3hNxkom1QPMb_IB-QVRACpscfNqZkSuMt0L1x6xmUNd_PfTSPh2hFkh5Ozs1WjtH-rfa0irU9W5FmfmHOMGb16rykwuapb6eop8B-UnDzLl45atdjAtPRcol21B3PyC0_GLoCeNRwm3Be7AEuxmXVVPXNkHHh6EHO4MkKixzq8lAWs2DoHPV9k67rk5m0SMRV8MAgrPYJeLGsrpv0UV5fneTLW9zGtkNnePcatrJ4kuwTh_rcFNXRsPEnUj7qjpWZ_E8nVEgkA7p390sejdjffwqBPh1aNtPbG-8XKFZmveZ0MTlHn-jox_USLIlSEn25gzoaeAkLW0lj6cTr9iXaNX9V2IgkfFLPz2NT5xM2FqijCO-6ZReBsfRc_KqRd53KZIbP0V6rvlZKeaT_pR8BNOs7BcUZDBganvRhFcUpLVVlN7uwgEktKnJRV6VkzMdS35lmfCMXTUjPowMSURNWsEE5r8BsrS0ueU2QReYvLFUG48BPe6et40cN5bW829q5W_hoP0mLdn-LsEN67eY0PAWMP0we24ICflOqE83_I78XVvt3PiXjtIFdHuv6Lc5Dg4mYoj9TQS83mgPd_JhhFUZO6bt3QWgtsxsoQK-FrIgI0nocVNPms_2jAwoAuSpfAh3q7wvNWBXyWeFNL7zOZe73q2xmYb8rxaxNX6HdJuIxDfyF2qLisnGDkxPhu3iDZxmJxQw9MqwAqCYeyW7xlOPY5BRsjP7kqakoMYsGU4KDcG1wy9fTSyKP952E-pMVlrsCIUieUrch0IUvTfo7oZNEc055Fx9d3CHE2gF3sak0DY3e1U8C0Buz2pKVD-AJ0TVP67zKxGTLE1fw7jumfpEWYsnV1gC6rcFYxY4RWFa8mzsi5yzuDMsJKz_mRmzDHkHGzmXUIrFvNS2huDR2b5T0D69FDWaFaoNX0C4KpBw8cwBueXNpLb2MzdINlC--ckPd86KWWMFrFiljI4IOtI2PrV7NlePgWOMypBBAYdTBo3Eb7gWsvdNKns0eibLqThECs3D3QUXXXjBArlfpp3VyqVCnjnR1J7eOW3TXi1BzXL65AbfoM-zlmpcmL9CmT3fXnlDZpvqIoG82EzwmJlgUtBz8JC19QdF4glXVRRujwUsOL7xhK_tSQJYr0ix5wQc0V1JDVp1RXnR9OoDBNtkw3AL_xYRhq-9vMd3tcS9MqIaTnsOXniLu7s-xyB71ayiDk_eSltKx64m05R80HvRnsQuGhO3thh8nW_Ch4eydhck-_5l-AJJ4yJtHrgOuLnvRDeplpF3kyx3npm7hiIqZDt2NixnN8kHeEeAwCzBJpmH-xPiPmF2Oc9BdONNs6Ja9O5TKbLYdnUx7PiHYGqHKYkjnPLFAVbo8sHnxo3fP6CsB9-mh8LHztSVaOi1iLS71lfIyxNdmkkgB17S2SKnmp3Dj_WSQc4otEfkpXDjXIeWnkjFfVyJw-aiteyFL3D_Q_7IvfY3deJw_UVTItJ1FvX6i0hrHej4YM-.wsX03Z3IHxbJ8l14O8tklw"

def audio_to_text(audio_message):
    model = whisper.load_model("base")
    result = model.transcribe(audio_message)
    return result["text"]


def get_chatgpt_response(message):
    # Get session _token F12/ inspect -> Storage -> Cookies -> secure..auth.sess.token -> value
    api = ChatGPT(session_token)
    # api2 = ChatGPT(email="akash.nagaraj1@gmail.com",password='password')
    # api3 = ChatGPT(session_token, conversation_id='some-random-uuid')

    response = api.send_message(message)
    print(response["message"])

    #api.refresh_auth() # refresh cauthorization token
    #api.reset_conversation() # reset conversation/ thread


def read_audio():
    import sounddevice as sd
    import numpy as np
    import scipy.io.wavfile as wav

    fs=44100
    duration = 5  # seconds
    myrecording = sd.rec(duration * fs, samplerate=fs, channels=2,dtype='float64')
    print("Recording Audio")
    sd.wait()
    print("Audio recording complete , Play Audio")
    sd.play(myrecording, fs)
    sd.wait()
    print("Play Audio Complete")


def main():
    """Get voice whisper to collect
    stable difusion image generation"""
    audio_message = read_audio()
    message = audio_to_text(audio_message)
    get_chatgpt_response(message)

if __name__=="__main__":
    main()
