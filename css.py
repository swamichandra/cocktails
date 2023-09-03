v1 = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Condensed&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans Condensed', sans-serif;
}

div.stButton > button {
                display: inline-block;
                outline: 0;
                cursor: pointer;
                border: none;
                padding: 0 56px;
                height: 45px;
                line-height: 45px;
                border-radius: 7px;
                background-color: #54442B;
                color: white;
                font-weight: 400;
                font-size: 16px;
                box-shadow: 0 4px 14px 0 rgb(0 118 255 / 39%);
                transition: background 0.2s ease,color 0.2s ease,box-shadow 0.2s ease;
                :hover{
                    color: #eee;
                    box-shadow: 0 6px 20px #54442B;
                }                       
}     

.small-font {
    font-size: 9px !important;
    font-weight : bold;
}



"""
