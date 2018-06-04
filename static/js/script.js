
var file1 = document.getElementById(`fileToUpload3`);
file1.onchange = function(){
    if(file1.files.length > 0)
    {

      document.getElementById(`fileToUpload3noFile`).innerHTML = file1.files[0].name;

    }
};
var file2 = document.getElementById(`fileToUpload2`);
file2.onchange = function(){
    if(file2.files.length > 0)
    {

      document.getElementById(`fileToUpload2noFile`).innerHTML = file2.files[0].name;

    }
};
var file3 = document.getElementById(`fileToUpload1`);
file3.onchange = function(){
    if(file3.files.length > 0)
    {

      document.getElementById(`fileToUpload1noFile`).innerHTML = file3.files[0].name;

    }
};
var file4 = document.getElementById(`fileToUpload4`);
file4.onchange = function(){
    if(file4.files.length > 0)
    {

      document.getElementById(`fileToUpload4noFile`).innerHTML = file4.files[0].name;

    }
};
