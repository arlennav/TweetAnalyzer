$('#myModal').modal({
  show: 'false'
}); 

$(document).ajaxStart(function () {
  $.LoadingOverlay("show", {
    background  : "rgba(29, 161, 242, 0.5)"
});
});

$(document).ajaxComplete(function () {
  $.LoadingOverlay("hide");
});


function sendTweet(tweetId){
  $.post('/api/tweet/'+ tweetId, {
  }, function (data) {
    $.LoadingOverlay("hide");
    $('#myModal').modal({
      show: 'true'
    }); 
  });
}