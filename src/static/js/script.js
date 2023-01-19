$(document).ready(function () {
    $('[data-toggle=offcanvas]').click(function () {
      $('.container').toggleClass('activate');
    });

    $('[data-toggle=ofset]').click(function () {
      $(".duo-wrapper").toggleClass("active");
    });

    $('[data-toggle=gradient-ofset]').click(function () {
      $(".gradient-wrapper").toggleClass("active");
    });

    $('[data-toggle=effect-ofset]').click(function () {
      $(".effect-wrapper").toggleClass("active");
    });

    const tabs = document.querySelectorAll('.menu_icons');
    tabs.forEach(clickedTab => {
      clickedTab.addEventListener('click',() => {
        tabs.forEach(tab => {
          tab.classList.remove('active');
        });
        clickedTab.classList.add('active')
      })
    })


    /*------------------imp tool for side_icons----------------------------------*/
    const enhance_icon = document.querySelectorAll('.enhance');
    const enhance_container = document.querySelectorAll(".enhance-container");

    const side_icon = document.querySelectorAll(".backg")
    const side_container = document.querySelectorAll(".side-container");

    $(enhance_icon).on("click",function(){
      enhance_container.forEach(tab => {
        /* tab.classList.toggle('visible'); */
        tab.classList.add("visible");
      });
      side_container.forEach(tab => {
        tab.classList.remove("visible");
      })
    })

    $(side_icon).on('click',function(){
      side_container.forEach(tab => {
        /* tab.classList.toggle("visible"); */
        tab.classList.add("visible");
      });
      enhance_container.forEach(tab=>{
        tab.classList.remove("visible");
      })
    })

    const button_icon = document.querySelectorAll(".fONwsr");
    button_icon.forEach(clickedTab => {
      clickedTab.addEventListener('click',() => {
        button_icon.forEach(tab => {
          tab.classList.remove('border_active');
        });
        clickedTab.classList.add('border_active')
      })
    })

   
    function createRipple(event) {
      const button = event.currentTarget;
      const circle = document.createElement("span");
      const diameter = Math.max(button.clientWidth, button.clientHeight);
      const radius = diameter / 2;
      circle.style.width = circle.style.height = `${diameter}px`;
      circle.style.left = `${event.clientX - (button.offsetLeft + radius)}px`;
      circle.style.top = `${event.clientY - (button.offsetTop + radius)}px`;
      circle.classList.add("ripples"); 

      const ripple = button.getElementsByClassName("ripples")[0];

      if (ripple) {
          ripple.remove();
      }
      button.appendChild(circle);
    }

  const buttons = document.getElementsByClassName("jWcpcT");
  for (const button of buttons) {
      button.addEventListener("click", createRipple);
  }

  const download_bots = document.getElementsByClassName("download_icon");
  for (const download_bot of download_bots){
    download_bot.addEventListener("click",createRipple);
  }

});