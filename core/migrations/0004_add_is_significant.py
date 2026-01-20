from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0003_forecasthistory_testhistory'),
    ]

    operations = [
        migrations.AddField(
            model_name='testhistory',
            name='is_significant',
            field=models.BooleanField(default=False),
        ),
    ]
