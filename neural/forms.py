from django.forms import forms, widgets


class New_DataSet(forms.Form):
    all_files = forms.FileField(widget=widgets.FileInput(attrs={'class': 'form-control form-control-lg', 'id': "formFileLg", 'multiple': True}))


class GetPhotoTest(forms.Form):
    file = forms.FileField(widget=widgets.FileInput(attrs={'class': 'form-control', 'id': "formFileMultiple"}))