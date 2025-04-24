import streamlit as st
import base64
import datetime as dt


###>>>---LOCAL FUNCTIONS---<<<###
def image_border_radius(image_path, border_radius, width, height, page_object = None, is_html = False):
    if is_html == False:
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        # Create HTML string with the image
        img_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="border-radius: {border_radius}px; width: {width}%; height: {height}%">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)
    else:
        # Create HTML string with the image
        img_html = f'<img src="{image_path}" style="border-radius: {border_radius}px; width: 300px;">'
        # Display the HTML string in Streamlit
        if page_object == None:
            st.markdown(img_html, unsafe_allow_html=True)
        else:
            page_object.markdown(img_html, unsafe_allow_html=True)

def timestamp_to_api_response(timestamp_date, timestamp_time):
    timestamp_string = " ".join([str(timestamp_date), str(timestamp_time)])
    input_format = "%Y-%m-%d %H:%M:%S"
    target_format = "%Y-%m-%dT%H:%M:%S.%f%z"
    timestamp_input = dt.datetime.strptime(timestamp_string, input_format)
    timestamp_target = timestamp_input.replace(tzinfo = dt.timezone.utc)
    timestamp = timestamp_target.strftime(target_format)
    return timestamp



###>>>---STREAMLIT FUNCTIONS---<<<###


###>>>---CACHE FUNCTION---<<<###
