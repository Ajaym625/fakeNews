from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('', views.index, name='index'),
    # path('news_url/',views.news_url,name='news_url'),
    path('analysis',views.analysis, name='analysis'),
    path('who',views.who, name='who'),
    path('when',views.when, name='when'),
    path('when_score',views.when_score, name='when_score'),
    path("what",views.what, name='what'),
    path('what_score',views.what_score, name='what_score'),
    path('where',views.where,name='where'),
    path('where_score',views.where_score, name='where_score'),
    path('why',views.why,name='why'),
    path('how',views.how,name='how'),
    path('final_analysis',views.final_analysis,name='final_analysis'),
    path('result',views.result, name='result')
]

if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)