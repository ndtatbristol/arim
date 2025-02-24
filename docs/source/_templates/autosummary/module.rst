{{ fullname }}
{{ "=" * fullname|length }}

.. automodule:: {{ fullname }}
   :no-members:

.. autosummary::
   :toctree: _autosummary/{{ fullname }}

   {% for item in classes %}
   {{ item }}
   {% endfor %}
   {% for item in functions %}
   {{ item }}
   {% endfor %}